"""Blogging adapter stub."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BlogPost:
    title: str
    body: str
    url: Optional[str] = None


class BloggingService:
    def __init__(self, publish_dir: str = "./blogs") -> None:
        self.publish_dir = publish_dir

    def draft(self, title: str, body: str) -> BlogPost:
        return BlogPost(title=title, body=body)

    def publish(self, post: BlogPost) -> BlogPost:
        post.url = f"{self.publish_dir}/{post.title.replace(' ', '-').lower()}.md"
        return post
