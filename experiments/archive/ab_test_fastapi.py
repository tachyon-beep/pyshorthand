"""FastAPI user management API.

Role: API
This module provides RESTful endpoints for user management.
"""

from functools import lru_cache

from pydantic import BaseModel


class User(BaseModel):
    """User model."""

    id: int
    name: str
    email: str
    active: bool = True


class UserAPI:
    """FastAPI user management routes."""

    def __init__(self, app):
        self.app = app
        self.users = {}
        self.next_id = 1

    @property
    def user_count(self) -> int:
        """Get total number of users. O(1)"""
        return len(self.users)

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format. O(1)"""
        return "@" in email and "." in email

    @lru_cache(maxsize=100)
    def get_user_by_email(self, email: str) -> User | None:
        """Get user by email with caching. O(N)"""
        for user in self.users.values():
            if user.email == email:
                return user
        return None


# Module-level API functions


def get_user(user_id: int) -> User | None:
    """Get user by ID.

    Time: O(1)
    """
    return users_db.get(user_id)


def create_user(name: str, email: str) -> User:
    """Create a new user.

    Complexity: O(1)
    """
    user = User(id=next_id(), name=name, email=email)
    users_db[user.id] = user
    return user


def update_user(
    user_id: int, name: str | None = None, email: str | None = None
) -> User | None:
    """Update user information. O(1)"""
    user = users_db.get(user_id)
    if user is None:
        return None

    if name:
        user.name = name
    if email:
        user.email = email

    return user


def delete_user(user_id: int) -> bool:
    """Delete user by ID. O(1)"""
    if user_id in users_db:
        del users_db[user_id]
        return True
    return False


def search_users(query: str) -> list[User]:
    """Search users by name or email.

    Runtime: O(N)
    """
    results = []
    query_lower = query.lower()

    for user in users_db.values():
        if query_lower in user.name.lower() or query_lower in user.email.lower():
            results.append(user)

    return results


# Global state
users_db = {}
_next_id = 1


def next_id() -> int:
    """Generate next user ID."""
    global _next_id
    current = _next_id
    _next_id += 1
    return current
