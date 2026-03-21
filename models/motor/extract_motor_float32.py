"""
Extract MOTOR representations using pure JAX fallbacks.
Monkeypatches to avoid missing CUDA custom kernels.
"""
import argparse
import datetime
import functools
import logging
import os
import pickle
import random
import tempfile
import time

import haiku as hk
import jax
import jax.numpy as jnp
import msgpack
import numpy as np

# MUST patch femr.jax BEFORE importing femr.models.transformer
import femr.jax as femr_jax

# ============================================================
# MONKEYPATCH 1: Replace gather_scatter_add with pure JAX
# ============================================================
def _pure_jax_gather_scatter(a, indices, output_dim):
    return femr_jax.gather_scatter_add_fallback(a, indices, output_dim)

femr_jax.gather_scatter_add = _pure_jax_gather_scatter

# ============================================================
# MONKEYPATCH 2: Replace local_attention with pure JAX fallback
# Signature: (queries, keys, values, length_mask, attention_width, causal=True) -> Array
# ============================================================
def _pure_jax_local_attention(queries, keys, values, length_mask, attention_width, causal=True):
    dummy_attn, result = femr_jax.local_attention_fallback(
        queries, keys, values, length_mask, attention_width, causal
    )
    return result

femr_jax.local_attention = _pure_jax_local_attention

logging.basicConfig(level=logging.INFO)
logging.info("Patched gather_scatter_add and local_attention to use pure JAX")

import femr.datasets
import femr.extension.dataloader
import femr.models.transformer

# ============================================================
# MONKEYPATCH 3: Prevent float16 cast in Transformer
# ============================================================
def _patched_transformer_call(self, batch, is_training):
    ages = batch["ages"]
    normed_ages = batch["normalized_ages"]

    if self.config.get("is_hierarchical"):
        e = self.embed.embeddings
        x = femr_jax.gather_scatter_add(e, batch["sparse_token_indices"], batch["ages"].shape[0])
    else:
        x = self.embed(batch["tokens"])

    if self.config.get("note_embedding_data"):
        note_embedding_matrix = batch["note_embedding_bytes"].view(dtype=jnp.float16).reshape(-1, 768)
        note_embedding = note_embedding_matrix.at[batch["tokens"]].get(mode="clip")
        x = jnp.where(batch["is_note_embedding"].reshape(-1, 1), note_embedding, x)

    dummy_values = jnp.ones((1, 1), dtype=x.dtype)
    x = jnp.where(batch["valid_tokens"].reshape((-1, 1)), x, dummy_values)
    x = self.in_norm(x)
    # Stay float32

    normed_ages = normed_ages.astype(x.dtype)

    if self.config["rotary"] == "global":
        pos_embed = femr.models.transformer.fixed_pos_embedding(ages, self.config["hidden_size"], x.dtype)
    elif self.config["rotary"] == "per_head":
        pos_embed = femr.models.transformer.fixed_pos_embedding(
            ages, self.config["hidden_size"] // self.config["n_heads"], x.dtype)
    elif self.config["rotary"] == "disabled":
        pos_embed = None
    else:
        raise RuntimeError("Invalid rotary")

    layer_rngs = jax.random.split(hk.next_rng_key(), len(self.lifted_params))
    all_params = [lifted(rng, x, normed_ages, pos_embed, batch, is_training)
                  for lifted, rng in zip(self.lifted_params, layer_rngs)]
    flattened = [jax.tree_util.tree_flatten(a) for a in all_params]
    all_flat, all_defs = zip(*flattened)
    assert all(all_defs[0] == a for a in all_defs)
    all_stacked = [jnp.stack(tuple(a[i] for a in all_flat)) for i in range(len(all_flat[0]))]
    all_stacked_tree = [jax.tree_util.tree_unflatten(all_defs[0], all_stacked), layer_rngs]

    def process(v, params_rng):
        params, rng = params_rng
        res = self.layer_transform.apply(params, rng, v, normed_ages, pos_embed, batch, is_training)
        return (v + res, None)

    final_x = jax.lax.scan(process, x, all_stacked_tree)[0]
    return self.out_norm(final_x)

femr.models.transformer.Transformer.__call__ = _patched_transformer_call
logging.info("Patched Transformer to stay in float32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=str)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prediction_times_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()

    with open(os.path.join(args.model_path, "model", "config.msgpack"), "rb") as f:
        config = msgpack.load(f, use_list=False)

    random.seed(config["seed"])
    config = hk.data_structures.to_immutable_dict(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        batches_path = os.path.join(tmpdir, "task_batches")

        command = f"clmbr_create_batches {batches_path} --data_path {args.data_path}"
        command += f" --task labeled_patients --labeled_patients_path {args.prediction_times_path} --val_start 70"
        command += f" --dictionary_path {args.model_path}/dictionary"
        if config["transformer"]["is_hierarchical"]:
            command += " --is_hierarchical"
        command += f" --transformer_vocab_size {config['transformer']['vocab_size']}"
        command += f" --batch_size {args.batch_size}"

        logging.info("Creating batches: %s", command)
        ret = os.system(command)
        if ret != 0:
            raise RuntimeError(f"clmbr_create_batches failed with exit code {ret}")

        batch_info_path = os.path.join(batches_path, "batch_info.msgpack")
        database = femr.datasets.PatientDatabase(args.data_path)

        with open(os.path.join(args.model_path, "model", "best"), "rb") as f:
            params = pickle.load(f)

        params = femr.models.transformer.convert_params(params, dtype=jnp.float32)
        logging.info("Loaded params in float32")

        with open(batch_info_path, "rb") as f:
            batch_info = msgpack.load(f, use_list=False)

        with open(os.path.join(args.model_path, "model", "config.msgpack"), "rb") as f:
            config = msgpack.load(f, use_list=False)
        config = hk.data_structures.to_immutable_dict(config)
        random.seed(config["seed"])
        rng = jax.random.PRNGKey(42)

        assert batch_info["config"]["task"]["type"] == "labeled_patients"
        loader = femr.extension.dataloader.BatchLoader(args.data_path, batch_info_path)

        n_train = loader.get_number_of_batches("train")
        n_dev = loader.get_number_of_batches("dev")
        n_test = loader.get_number_of_batches("test")
        total_batches = n_train + n_dev + n_test
        logging.info("Batches: train=%d, dev=%d, test=%d (total=%d)", n_train, n_dev, n_test, total_batches)

        def model_fn(config, batch):
            return femr.models.transformer.EHRTransformer(config)(batch, no_task=True)

        model = hk.transform(model_fn)

        @functools.partial(jax.jit, static_argnames=("config"))
        def compute_repr(params, rng, config, batch):
            repr, mask = model.apply(params, rng, config, batch)
            offsets = jnp.ones((repr.shape[0], 1), dtype=repr.dtype)
            return jnp.concatenate((repr, offsets), axis=-1)

        l_reprs = []
        l_repr_ages = []
        l_repr_pids = []
        l_repr_offsets = []
        total_done = 0
        t0 = time.time()

        for i, split in enumerate(("train", "dev", "test")):
            n_batches = loader.get_number_of_batches(split)
            logging.info("Processing split=%s, %d batches", split, n_batches)

            for j in range(n_batches):
                raw_batch = loader.get_batch(split, j)
                batch = jax.tree_map(lambda a: jax.device_put(a), raw_batch)

                repr = compute_repr(params, rng, config, batch)

                def slice_fn(val):
                    if len(val.shape) == 3:
                        return val[:batch["num_indices"], :, :]
                    elif len(val.shape) == 2:
                        return val[:batch["num_indices"], :]
                    elif len(val.shape) == 1:
                        return val[:batch["num_indices"]]

                p_index = batch["transformer"]["label_indices"] // batch["transformer"]["length"]
                p_index = slice_fn(p_index)

                l_reprs.append(np.array(slice_fn(repr)))
                l_repr_ages.append(
                    raw_batch["transformer"]["integer_ages"][
                        np.array(slice_fn(batch["transformer"]["label_indices"]))
                    ]
                )
                l_repr_pids.append(raw_batch["patient_ids"][np.array(p_index)])
                l_repr_offsets.append(raw_batch["offsets"][np.array(p_index)])

                total_done += 1
                if total_done <= 5 or total_done % 100 == 0:
                    elapsed = time.time() - t0
                    rate = total_done / elapsed if elapsed > 0 else 0
                    eta = (total_batches - total_done) / rate if rate > 0 else 0
                    logging.info("  batch %d/%d (%.1f b/s, ETA %.0fs)",
                                 total_done, total_batches, rate, eta)

        elapsed = time.time() - t0
        logging.info("Inference done in %.1fs (%.1f batches/s)", elapsed, total_done/elapsed)
        logging.info("Concatenating results...")

        all_reprs = np.concatenate(l_reprs, axis=0)
        all_ages = np.concatenate(l_repr_ages, axis=0)
        all_pids = np.concatenate(l_repr_pids, axis=0)
        all_offsets = np.concatenate(l_repr_offsets, axis=0)

        logging.info("Representations shape: %s", all_reprs.shape)

        prediction_times = []
        for pid, age, offset in zip(all_pids, all_ages, all_offsets):
            birth_date = datetime.datetime.combine(
                database.get_patient_birth_date(int(pid)), datetime.time.min
            )
            pred_time = birth_date + datetime.timedelta(minutes=int(age))
            prediction_times.append(pred_time)

        result = {
            "data_path": args.data_path,
            "model": args.model_path,
            "representations": all_reprs,
            "patient_ids": all_pids,
            "prediction_times": np.array(prediction_times),
            "label_ages": all_ages,
        }

        with open(args.destination, "wb") as wf:
            pickle.dump(result, wf)

        logging.info("Saved %d representations (%s) to %s",
                      all_reprs.shape[0], all_reprs.shape, args.destination)


if __name__ == "__main__":
    main()
