# -*- coding: utf-8 -*-
"""ctypes bindings for llama.cpp -- only this layer touches the DLL, the two above never do.

Struct names are one word where possible: Mp / Cp / Msg / Batch (no ModelParams / ContextParams).
Field names were already short, so keep them: ctypes aligns by position+type, a rename changes nothing.
"""
import ctypes as ct
import os

DLL = r"" #DLL
GGUF = r"" #GGUF
Q4_0 = 2   # KV quantization type. Also the source of the measured "engine output depends on how the KV was built"


class Mp(ct.Structure):
    _fields_ = [("d", ct.c_void_p), ("t", ct.c_void_p), ("ngl", ct.c_int32),
                ("sm", ct.c_int), ("lm", ct.c_int), ("lz", ct.c_int), ("mg", ct.c_int32),
                ("ts", ct.c_void_p), ("pc", ct.c_void_p), ("pcu", ct.c_void_p),
                ("k", ct.c_void_p), ("vo", ct.c_bool), ("ct", ct.c_bool), ("u", ct.c_bool),
                ("nh", ct.c_bool), ("na", ct.c_bool), ("mtp", ct.c_bool)]


class Cp(ct.Structure):
    _fields_ = [("n_ctx", ct.c_uint32), ("n_batch", ct.c_uint32), ("n_ubatch", ct.c_uint32),
                ("n_seq_max", ct.c_uint32), ("n_rs_seq", ct.c_uint32),
                ("n_outputs_max", ct.c_uint32), ("nomps", ct.c_uint32),
                ("nt", ct.c_int32), ("ntb", ct.c_int32), ("cty", ct.c_int),
                ("rs", ct.c_int), ("pt", ct.c_int), ("at", ct.c_int), ("fat", ct.c_int),
                ("rfb", ct.c_float), ("rfs", ct.c_float), ("yef", ct.c_float),
                ("yaf", ct.c_float), ("ybf", ct.c_float), ("ybs", ct.c_float),
                ("yoc", ct.c_uint32), ("dt", ct.c_float), ("cb", ct.c_void_p),
                ("cbu", ct.c_void_p), ("tk", ct.c_int), ("tv", ct.c_int),
                ("ac", ct.c_void_p), ("acd", ct.c_void_p), ("emb", ct.c_bool),
                ("okq", ct.c_bool), ("np", ct.c_bool), ("oo", ct.c_bool),
                ("swaf", ct.c_bool), ("kvu", ct.c_bool), ("smp", ct.c_void_p),
                ("nsmp", ct.c_size_t), ("co", ct.c_void_p)]


class Msg(ct.Structure):
    _fields_ = [("role", ct.c_char_p), ("content", ct.c_char_p)]


class Batch(ct.Structure):
    _fields_ = [("n_tokens", ct.c_int32), ("token", ct.POINTER(ct.c_int32)),
                ("embd", ct.POINTER(ct.c_float)), ("pos", ct.POINTER(ct.c_int32)),
                ("n_seq_id", ct.POINTER(ct.c_int32)),
                ("seq_id", ct.POINTER(ct.POINTER(ct.c_int32))),
                ("logits", ct.POINTER(ct.c_int8))]


def bind():
    """Load the DLL and pin down every function signature we need in one pass.

    The trap with incomplete signatures: with no argtypes declared, llama_token_to_piece
    makes ctypes truncate 64-bit pointers to int, raising
    OverflowError: int too long to convert (hit this for real).
    """
    g = ct.CDLL(os.path.join(DLL, "ggml.dll"), mode=ct.RTLD_GLOBAL)
    L = ct.CDLL(os.path.join(DLL, "llama.dll"))
    # Route llama.cpp's own logging through ours BEFORE the backend loads.  By
    # default it writes to stderr, which on this model means every decode prints
    # CUDA graph lines -- tens of thousands a session -- and the messages that
    # matter drown.  Installing the callback after this point would miss the
    # whole backend and model load.
    try:
        import log as _log
        gb = None
        for name in ("ggml-base.dll", "ggml.dll"):
            try:
                gb = ct.CDLL(os.path.join(DLL, name))
                break
            except OSError:
                continue
        _log.register("llama", L)
        _log.register("ggml-base", gb)
        _log.capture_llama_log()
    except Exception:
        pass          # logging is never worth failing a load over
    g.ggml_backend_load_all_from_path.restype = None
    g.ggml_backend_load_all_from_path.argtypes = [ct.c_char_p]
    g.ggml_backend_load_all_from_path(DLL.encode())

    L.llama_backend_init.restype = None
    L.llama_backend_init()
    L.llama_model_default_params.restype = Mp
    L.llama_context_default_params.restype = Cp
    L.llama_model_load_from_file.restype = ct.c_void_p
    L.llama_model_load_from_file.argtypes = [ct.c_char_p, Mp]
    L.llama_init_from_model.restype = ct.c_void_p
    L.llama_init_from_model.argtypes = [ct.c_void_p, Cp]
    L.llama_model_get_vocab.restype = ct.c_void_p
    L.llama_model_get_vocab.argtypes = [ct.c_void_p]
    L.llama_vocab_n_tokens.restype = ct.c_int32
    L.llama_vocab_n_tokens.argtypes = [ct.c_void_p]
    L.llama_tokenize.restype = ct.c_int32
    L.llama_tokenize.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_int32,
                                 ct.POINTER(ct.c_int32), ct.c_int32, ct.c_bool, ct.c_bool]
    L.llama_token_to_piece.restype = ct.c_int32
    L.llama_token_to_piece.argtypes = [ct.c_void_p, ct.c_int32, ct.c_char_p,
                                       ct.c_int32, ct.c_int32, ct.c_bool]
    L.llama_batch_get_one.restype = Batch
    L.llama_batch_get_one.argtypes = [ct.POINTER(ct.c_int32), ct.c_int32]
    L.llama_decode.restype = ct.c_int32
    L.llama_decode.argtypes = [ct.c_void_p, Batch]
    L.llama_get_logits.restype = ct.POINTER(ct.c_float)
    L.llama_get_logits.argtypes = [ct.c_void_p]
    L.llama_get_logits_ith.restype = ct.POINTER(ct.c_float)
    L.llama_get_logits_ith.argtypes = [ct.c_void_p, ct.c_int32]
    L.llama_get_memory.restype = ct.c_void_p
    L.llama_get_memory.argtypes = [ct.c_void_p]
    L.llama_memory_seq_rm.restype = ct.c_bool
    L.llama_memory_seq_rm.argtypes = [ct.c_void_p, ct.c_int32, ct.c_int32, ct.c_int32]
    L.llama_memory_seq_pos_max.restype = ct.c_int32
    L.llama_memory_seq_pos_max.argtypes = [ct.c_void_p, ct.c_int32]
    L.llama_memory_seq_cp.restype = None
    L.llama_memory_seq_cp.argtypes = [ct.c_void_p, ct.c_int32, ct.c_int32,
                                      ct.c_int32, ct.c_int32]
    L.llama_model_meta_val_str.restype = ct.c_int32
    L.llama_model_meta_val_str.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_char_p, ct.c_size_t]
    L.llama_chat_apply_template.restype = ct.c_int32
    L.llama_chat_apply_template.argtypes = [ct.c_char_p, ct.POINTER(Msg), ct.c_size_t,
                                            ct.c_bool, ct.c_char_p, ct.c_int32]

    # Logging callback -- suppress llama.cpp C-level logs (slot, CUDA graph, etc.)
    L._logcb = ct.CFUNCTYPE(None, ct.c_int, ct.c_char_p, ct.c_void_p)
    L._logcb_ref = L._logcb(lambda _lvl, _txt, _ud: None)
    L.llama_log_set.argtypes = [L._logcb, ct.c_void_p]
    L.llama_log_set.restype = None
    L.llama_log_set(L._logcb_ref, None)
    return L
