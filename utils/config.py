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