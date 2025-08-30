import os, json

# utils/config.py 之类
def args_to_config(args):
    return {
        "general": {
            "agent_name": args.agent_name,
            "log_dir": args.log_dir,
            "log_filename": args.log_filename,
        },
        "agent": {
            "agent_1": {
                "embedding": {
                    "detail": {
                        "embedding_model": args.embedding_model,
                        "chunk_size": args.chunk_size,
                        "verbose": bool(getattr(args, "verbose", False)),
                        "openai_api_key": getattr(args, "openai_api_key", None),
                    }
                }
            }
        },
        "short": {
            "jump_threshold_upper": args.short_jump_threshold_upper,
            "importance_score_initialization": args.short_importance_score_initialization,
            "decay_params": {
                "recency_factor": args.short_recency_factor,
                "importance_factor": args.short_importance_factor,
            },
            "clean_up_threshold_dict": {
                "recency_threshold": args.short_recency_threshold,
                "importance_threshold": args.short_importance_threshold,
            },
        },
        "mid": {
            "jump_threshold_upper": args.mid_jump_threshold_upper,
            "jump_threshold_lower": args.mid_jump_threshold_lower,
            "importance_score_initialization": args.mid_importance_score_initialization,
            "decay_params": {
                "recency_factor": args.mid_recency_factor,
                "importance_factor": args.mid_importance_factor,
            },
            "clean_up_threshold_dict": {
                "recency_threshold": args.mid_recency_threshold,
                "importance_threshold": args.mid_importance_threshold,
            },
        },
        "long": {
            "jump_threshold_lower": args.long_jump_threshold_lower,
            "importance_score_initialization": args.long_importance_score_initialization,
            "decay_params": {
                "recency_factor": args.long_recency_factor,
                "importance_factor": args.long_importance_factor,
            },
            "clean_up_threshold_dict": {
                "recency_threshold": args.long_recency_threshold,
                "importance_threshold": args.long_importance_threshold,
            },
        },
        "reflection": {
            "importance_score_initialization": args.reflection_importance_score_initialization,
            "decay_params": {
                "recency_factor": args.reflection_recency_factor,
                "importance_factor": args.reflection_importance_factor,
            },
            "clean_up_threshold_dict": {
                "recency_threshold": args.reflection_recency_threshold,
                "importance_threshold": args.reflection_importance_threshold,
            },
        },
    }


def load_existing_keys(jsonl_path):
    """从已存在的 jsonl 文件中读取所有 (ticker|date) 唯一键"""
    keys = set()
    if not os.path.exists(jsonl_path):
        return keys
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                meta = obj.get("meta", {})
                k = f"{meta.get('ticker','')}|{meta.get('date','')}"
                if "|" in k.strip("|"):
                    keys.add(k)
            except Exception:
                # 如果存在脏行，直接跳过，避免中断
                continue
    return keys

def append_jsonl_once(jsonl_path, record, key, seen_keys):
    """仅当 key 未出现过时追加写入，并将 key 加入 seen_keys"""
    if key in seen_keys:
        return False
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    seen_keys.add(key)
    return True