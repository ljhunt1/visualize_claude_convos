"""Run the whole pipeline in order. Each stage skips itself when fresh.

Usage: uv run --env-file .env python pipeline/run_all.py
"""
import asyncio

import s1_extract
import s2_label
import s3_vectorize
import s4_landmarks
import s5_project
import s6_cluster
import s7_name_clusters
import s8_export


def main() -> None:
    stages = [
        ("s1_extract", s1_extract.main),
        ("s2_label", lambda: asyncio.run(s2_label.main())),
        ("s3_vectorize", s3_vectorize.main),
        ("s4_landmarks", s4_landmarks.main),
        ("s5_project", s5_project.main),
        ("s6_cluster", s6_cluster.main),
        ("s7_name_clusters", s7_name_clusters.main),
        ("s8_export", s8_export.main),
    ]
    for name, fn in stages:
        print(f"\n=== {name} ===")
        fn()


if __name__ == "__main__":
    main()
