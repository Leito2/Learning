# 🛡️ RBAC, Scopes, and ABAC

## 🎯 Learning Objectives

- Distinguish Role-Based Access Control (RBAC), Scope-based authz, and Attribute-Based Access Control (ABAC)
- Implement RBAC in FastAPI with role-checking dependencies
- Use OAuth2 scopes with `SecurityScopes` for fine-grained permissions
- Build ABAC for row-level permissions (e.g., "users can only edit their own posts")
- Avoid the five most common authorization mistakes in production

## Introduction

Authentication answers "who is this user?"; authorization answers "what can this user do?" A complete security model needs both. The previous notes covered authentication: passwords, tokens, OAuth2 flows. This note covers the other half.

The three main authorization models are **RBAC** (roles), **scope-based** (permissions on tokens), and **ABAC** (attribute-based, for fine-grained rules). Each solves a different problem. RBAC is intuitive ("admin can do anything; members can read; viewers can read public content") but coarse. Scopes are good for API permissions ("this token can read users but not delete them"). ABAC handles complex rules ("a user can edit a post only if they are the author AND the post is not locked").

The right choice depends on the question being asked. RBAC answers "what is your role?" Scopes answer "what is your token allowed to do?" ABAC answers "given everything we know about you, this resource, and the action, is this allowed?" Most systems use a mix.

This note covers all three, with FastAPI patterns for each. The capstone at the end of the course combines them.

---

## 1. RBAC: Role-Based Access Control

### 1.1 The model

A user has **roles** (e.g., `"admin"`, `"member"`, `"viewer"`). Each role grants a set of **permissions** (e.g., `read:users`, `write:posts`). Permissions are checked at the action level.

```python
# Static role-permission mapping
ROLE_PERMISSIONS = {
    "admin": {
        "read:users", "write:users", "delete:users",
        "read:posts", "write:posts", "delete:posts",
        "read:billing", "write:billing",
    },
    "member": {
        "read:users",
        "read:posts", "write:posts",  # can edit own posts only (ABAC)
    },
    "viewer": {
        "read:posts",
    },
}


def has_permission(user: User, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(user.role, set())
```

### 1.2 The FastAPI pattern

```python
from typing import Iterable
from fastapi import Depends, HTTPException, status


def require_role(*allowed_roles: str):
    """Dependency factory: user must have one of the allowed roles."""
    async def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' is not allowed; need one of {allowed_roles}",
            )
        return user
    return checker


def require_permission(*required_permissions: str):
    """Dependency factory: user must have all required permissions."""
    async def checker(user: User = Depends(get_current_user)) -> User:
        for perm in required_permissions:
            if perm not in ROLE_PERMISSIONS.get(user.role, set()):
                raise HTTPException(
                    status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{perm}' required",
                )
        return user
    return checker


# Usage
@router.get("/admin/users")
async def list_all_users(user: User = Depends(require_role("admin"))):
    return await uow.users.list()


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(
    user_id: int,
    user: User = Depends(require_permission("delete:users")),
):
    ...
```

The `require_role` and `require_permission` functions are **dependency factories** — they return a dependency function. This pattern lets handlers declare their access requirements inline.

### 1.3 Roles vs permissions

Two common designs:

| Design | Example | Pros | Cons |
|--------|---------|------|------|
| Role-only | `user.role == "admin"` | Simple, no mapping table | Adding a new permission requires changing code |
| Role + permissions | `user.role = "admin"`, has `read:users`, `write:users` | Permissions can be added without changing code | Needs a permissions table |

The role + permissions design scales better. A new permission can be granted to existing roles without code changes. The mapping is in the database, not in code:

```python
class Role(Base):
    __tablename__ = "roles"
    name: Mapped[str] = mapped_column(String(50), primary_key=True)
    permissions: Mapped[list[str]] = mapped_column(JSON)


class User(Base):
    role: Mapped[str] = mapped_column(ForeignKey("roles.name"))
```

For most apps, the static `ROLE_PERMISSIONS` dict is enough. The database-driven approach is needed only when admins need to edit roles at runtime.

---

## 2. Scope-Based Authorization (OAuth2 Scopes)

### 2.1 The model

OAuth2 tokens carry **scopes** — strings that describe what the token can do. The client requests scopes during the auth flow; the auth server issues a token with the approved scopes. The resource server checks the scopes on every request.

```
Client requests:    scope="read:users write:posts"
User approves:      scope="read:users write:posts"  (or a subset)
Token issued:       scope="read:users write:posts"
Resource check:     "read:users" in token.scope
```

### 2.2 The FastAPI pattern

FastAPI's `SecurityScopes` is the canonical way to enforce scopes:

```python
from fastapi import Depends, Security, HTTPException
from fastapi.security import SecurityScopes
from fastapi.security.api_key import APIKeyHeader


async def get_current_token(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme),
) -> dict:
    """Validate the token and check that all required scopes are present."""
    # In production: validate the JWT
    payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
    # Check scopes
    token_scopes = payload.get("scopes", [])
    missing_scopes = [scope for scope in security_scopes.scopes if scope not in token_scopes]
    if missing_scopes:
        raise HTTPException(
            403,
            detail=f"Missing required scopes: {missing_scopes}",
            headers={"WWW-Authenticate": f'Bearer scope="{security_scopes.scope_str}"'},
        )
    return payload


@app.get("/users")
async def list_users(
    token: dict = Security(get_current_token, scopes=["read:users"]),
):
    ...


@app.delete("/users/{user_id}", status_code=204)
async def delete_user(
    user_id: int,
    token: dict = Security(get_current_token, scopes=["read:users", "write:users"]),
):
    ...
```

The `Security` function (vs `Depends`) is for declaring additional security scopes. The `security_scopes.scope_str` returns the OAuth2-formatted scope string for the `WWW-Authenticate` header.

### 2.3 Where scopes come from

In an OAuth2 setup, scopes are issued by the auth server and stored in the access token:

```python
# Token issuance
def create_access_token(*, sub: int, scopes: list[str]) -> str:
    payload = {
        "sub": str(sub),
        "scopes": scopes,
        "exp": int(time.time()) + 900,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


# Resource server (FastAPI)
async def get_current_token(...):
    payload = jwt.decode(token, ...)
    return {"user_id": payload["sub"], "scopes": payload.get("scopes", [])}
```

For a non-OAuth2 service, scopes can be derived from the user's role at login time:

```python
async def login(email: str, password: str) -> dict:
    user = await authenticate(email, password)
    role_scopes = ROLE_SCOPES[user.role]  # admin → ["read:users", "write:users", ...]
    token = create_access_token(sub=user.id, scopes=role_scopes)
    return {"access_token": token}
```

---

## 3. ABAC: Attribute-Based Access Control

### 3.1 The model

RBAC and scopes answer "what is your role/permission?" ABAC answers "given everything we know, is this allowed?" ABAC uses **policies** that evaluate attributes of the user, the resource, and the action.

```
Policy: "A user can edit a post if they are the author AND the post is not locked."

Attributes:
  user.id: 42
  user.role: "member"
  post.author_id: 42
  post.is_locked: false
  action: "edit"

Evaluation:
  user.id == post.author_id → true
  post.is_locked == false → true
  Allow.
```

### 3.2 The FastAPI pattern

ABAC is best implemented as a **policy** class or function. The handler delegates the decision to the policy.

```python
from typing import Protocol


class AuthorizationPolicy(Protocol):
    def can_edit_post(self, user: User, post: Post) -> bool: ...
    def can_delete_user(self, user: User, target: User) -> bool: ...


class DefaultPolicy:
    def can_edit_post(self, user: User, post: Post) -> bool:
        if user.role == "admin":
            return True
        return post.author_id == user.id and not post.is_locked

    def can_delete_user(self, user: User, target: User) -> bool:
        # Only admins can delete users
        return user.role == "admin" and user.id != target.id  # can't delete self


# Dependency that returns the policy
def get_policy() -> AuthorizationPolicy:
    return DefaultPolicy()


# Handler
@router.patch("/posts/{post_id}")
async def update_post(
    post_id: int,
    payload: PostUpdate,
    uow: UnitOfWork = Depends(get_uow),
    user: User = Depends(get_current_user),
    policy: AuthorizationPolicy = Depends(get_policy),
):
    post = await uow.posts.get(post_id)
    if not post:
        raise HTTPException(404, "Post not found")
    if not policy.can_edit_post(user, post):
        raise HTTPException(403, "You cannot edit this post")
    # Apply the update
    post = await uow.posts.update(post, **payload.model_dump(exclude_unset=True))
    await uow.commit()
    return post
```

The policy is injected, not hardcoded. Tests can override the policy; different deployments can use different policies.

### 3.3 Policy libraries

For complex policies, a library like `casbin` or `opal` provides a policy engine:

```python
import casbin


e = casbin.Enforcer("path/to/model.conf", "path/to/policy.csv")

# Check permission
if e.enforce(user_id, post_id, "edit"):
    ...
```

`casbin` policies are written in a DSL-like syntax and stored in files or a database. For most FastAPI services, the Python policy class is simpler; `casbin` shines when policies need to be edited at runtime by non-developers.

---

## 4. Row-Level Security (RLS) for ABAC

### 4.1 The pattern

For data-level ABAC ("a user can only see their own row"), the database can enforce the rule. PostgreSQL Row-Level Security policies filter rows based on the session's `current_setting`:

```sql
-- A user can see only their own posts
CREATE POLICY user_owns_post ON posts
    USING (author_id = current_setting('app.current_user_id')::INTEGER);
```

The application sets the session variable:

```sql
SET LOCAL app.current_user_id = 42;
```

A query:

```sql
SELECT * FROM posts;
```

Returns only the posts where `author_id = 42`. The application code never has to filter; the database does it.

This is covered in detail in [[../38 - SQLAlchemy 2.0 Async + Alembic for FastAPI/06 - Multi-Tenancy Patterns|the multi-tenancy note]] (note 06 of the SQLAlchemy course). The same pattern applies to row-level ABAC.

### 4.2 When to use RLS vs application-level checks

| Use RLS when | Use application-level when |
|--------------|----------------------------|
| The rule is "this row belongs to user X" | The rule is complex (multiple conditions) |
| The rule is stable per row | The rule changes per request |
| The data layer is the source of truth | The decision needs context not in the DB |
| You have multiple services reading the same data | The decision is product-specific |

The "this row belongs to user X" rule is the most common ABAC; RLS is the right tool. Complex rules belong in the application.

---

## 5. Combining RBAC, Scopes, and ABAC

### 5.1 The layered model

Most production systems use all three:

```
Request arrives
  ↓
Authentication: who is this user?
  ↓
Scope check: does the token have the required scopes for this endpoint?
  ↓
RBAC check: does the user's role grant access to this endpoint?
  ↓
ABAC check: given this user and this resource, is the action allowed?
  ↓
Response
```

Scopes are checked first because they are the cheapest. RBAC is checked next. ABAC is checked last because it requires loading the resource.

### 5.2 A unified permission decorator

```python
from functools import wraps
from typing import Callable


def require(
    *,
    scopes: list[str] | None = None,
    roles: list[str] | None = None,
    abac: Callable | None = None,
) -> Callable:
    """Combined permission check: scopes, then roles, then ABAC."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 1) Scopes (from the token)
            token = kwargs.get("token", {})
            for scope in scopes or []:
                if scope not in token.get("scopes", []):
                    raise HTTPException(403, f"Missing scope: {scope}")
            # 2) Roles (from the user)
            user = kwargs.get("user")
            if roles and user.role not in roles:
                raise HTTPException(403, f"Role {user.role} not allowed")
            # 3) ABAC (custom check)
            if abac and not abac(user, kwargs.get("resource")):
                raise HTTPException(403, "ABAC policy denied")
            return await func(*args, **kwargs)
        return wrapper
    return decorator


@router.delete("/users/{user_id}", status_code=204)
@require(scopes=["write:users"], roles=["admin"], abac=policy.can_delete_user)
async def delete_user(
    user_id: int,
    user: User = Depends(get_current_user),
    token: dict = Security(get_current_token, scopes=["write:users"]),
    ...
):
    ...
```

The decorator reads the dependencies from `kwargs` and checks them in order. A missing scope fails first (cheapest); a missing role fails next; a failed ABAC fails last (requires the resource to be loaded).

### 5.3 The Open Policy Agent (OPA) for complex policies

For services with many ABAC rules that change often, Open Policy Agent (OPA) is the industry standard:

```rego
# policy.rego
package authz

default allow = false

allow {
    input.user.role == "admin"
}

allow {
    input.action == "edit"
    input.user.id == input.resource.author_id
    not input.resource.is_locked
}
```

The service sends `(user, action, resource)` to OPA, which returns `true`/`false`. The service never holds the policy in code. This pattern is standard in microservices.

---

## 6. The Five Authorization Pitfalls

### 6.1 Checking role on the client

```python
# ❌ Frontend hides the delete button; backend doesn't check
@router.delete("/users/{user_id}")
async def delete_user(user_id: int, user: User = Depends(get_current_user)):
    # No role check!
    await uow.users.delete(user_id)
```

The client-side check is for UX. The server-side check is for security. A user who reverse-engineers the API can call the endpoint directly; without a server-side check, the action is allowed.

### 6.2 Using the user's role in the JWT, not the DB

```python
# ❌ Stale role in the token
payload = {"sub": user.id, "role": "admin"}  # captured at login
# Later: user.role changed to "member" in the DB
# The token still says "admin" until it expires
```

For long-lived tokens, the role check should always query the DB. The token carries identity (`sub`), not current permissions.

### 6.3 Confusing authentication and authorization

```python
# ❌ The handler is "authenticated" but not "authorized"
@router.get("/admin")
async def admin_panel(user: User = Depends(get_current_user)):
    # No role check; any authenticated user can access
    return {"sensitive": "data"}
```

`get_current_user` proves who the user is. It does not prove they are allowed to access this endpoint. The endpoint must check authorization explicitly.

### 6.4 Hardcoding role names

```python
# ❌ The role name is a string literal everywhere
if user.role == "admin":
    ...
if user.role == "admin":  # typo: "admiin"
    ...
```

Define role names as constants:

```python
# app/core/roles.py
class Role:
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"
    SYSTEM = "system"  # internal, non-human


ROLE_PERMISSIONS = {
    Role.ADMIN: {...},
    Role.MEMBER: {...},
    Role.VIEWER: {...},
    Role.SYSTEM: {"*"},  # all permissions
}
```

A typo becomes a `NameError` at import time, not a silent authorization bypass in production.

### 6.5 Trusting the resource's `owner_id` from the client

```python
# ❌ Client sets the owner_id
@router.post("/posts")
async def create_post(
    payload: PostCreate,  # includes owner_id from the client
    user: User = Depends(get_current_user),
):
    post = await uow.posts.create(**payload.model_dump(), owner_id=payload.owner_id)
    # The user can set any owner_id, including someone else's
```

The server must set ownership from the authenticated identity, never from the client request:

```python
# ✅ Server sets owner_id from the authenticated user
post = await uow.posts.create(
    **payload.model_dump(),
    owner_id=user.id,  # not from the client
    tenant_id=user.tenant_id,
)
```

---

## 7. Testing Authorization

### 7.1 Test the role checks

```python
@pytest.mark.asyncio
async def test_admin_endpoint_rejects_member(client, member_user):
    token = create_access_token(sub=member_user.id, role="member")
    response = await client.get(
        "/admin/users",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_admin_endpoint_allows_admin(client, admin_user):
    token = create_access_token(sub=admin_user.id, role="admin")
    response = await client.get(
        "/admin/users",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
```

### 7.2 Test the ABAC policies

```python
@pytest.mark.asyncio
async def test_user_can_edit_own_post(uow, user, post):
    policy = DefaultPolicy()
    assert policy.can_edit_post(user, post) is True


@pytest.mark.asyncio
async def test_user_cannot_edit_locked_post(uow, user, post):
    post.is_locked = True
    policy = DefaultPolicy()
    assert policy.can_edit_post(user, post) is False


@pytest.mark.asyncio
async def test_user_cannot_edit_other_user_post(uow, user, other_post):
    policy = DefaultPolicy()
    assert policy.can_edit_post(user, other_post) is False
```

### 7.3 Test the scope checks

```python
@pytest.mark.asyncio
async def test_endpoint_rejects_token_without_scope(client, user):
    # Issue a token without the required scope
    token = create_access_token(sub=user.id, scopes=["read:posts"])
    response = await client.delete(
        "/users/1",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403
    assert "write:users" in response.json()["detail"]
```

---

## 8. Código de Compresión

```python
"""
Compresión: RBAC, Scopes, and ABAC
Covers: role checking, scope-based authz, attribute-based policies,
       combining all three.
"""
from functools import wraps
from typing import Callable
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import SecurityScopes, OAuth2PasswordBearer
import jwt


# 1) Role and permission definitions
class Role:
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"
    SYSTEM = "system"


ROLE_PERMISSIONS = {
    Role.ADMIN: {"read:users", "write:users", "delete:users", "read:posts", "write:posts"},
    Role.MEMBER: {"read:users", "read:posts", "write:posts"},  # write:posts is ABAC-checked
    Role.VIEWER: {"read:posts"},
    Role.SYSTEM: {"*"},
}


# 2) Role check dependency factory
def require_role(*allowed_roles: str) -> Callable:
    async def checker(user = Depends(get_current_user)):
        if user.role not in allowed_roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Role {user.role} not in {allowed_roles}")
        return user
    return checker


# 3) Permission check dependency factory
def require_permission(*required: str) -> Callable:
    async def checker(user = Depends(get_current_user)):
        perms = ROLE_PERMISSIONS.get(user.role, set())
        if "*" in perms:
            return user
        for p in required:
            if p not in perms:
                raise HTTPException(status.HTTP_403_FORBIDDEN, f"Permission {p} required")
        return user
    return checker


# 4) Scope check
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_token_with_scopes(security_scopes: SecurityScopes, token: str = Depends(oauth2_scheme)):
    payload = jwt.decode(token, "secret", algorithms=["HS256"])
    scopes = payload.get("scopes", [])
    missing = [s for s in security_scopes.scopes if s not in scopes]
    if missing:
        raise HTTPException(403, detail=f"Missing scopes: {missing}")
    return payload


# 5) ABAC policy
class Policy:
    def can_edit_post(self, user, post) -> bool:
        if user.role == Role.ADMIN:
            return True
        return post.author_id == user.id and not post.is_locked


def get_policy() -> Policy:
    return Policy()


# 6) Combined decorator
def require(*, scopes=None, roles=None, abac_fn=None):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            token = kwargs.get("token", {})
            for s in scopes or []:
                if s not in token.get("scopes", []):
                    raise HTTPException(403, f"Missing scope: {s}")
            user = kwargs.get("user")
            if roles and user.role not in roles:
                raise HTTPException(403, f"Role {user.role} not allowed")
            if abac_fn and not abac_fn(user, kwargs.get("resource")):
                raise HTTPException(403, "ABAC denied")
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# 7) Demo handlers
from fastapi import APIRouter
router = APIRouter()


@router.get("/admin/users")
async def list_users(user=Depends(require_role(Role.ADMIN))):
    return {"users": []}


@router.delete("/users/{user_id}", status_code=204)
@require(scopes=["write:users"], roles=[Role.ADMIN], abac_fn=Policy().can_delete_user)
async def delete_user(
    user_id: int,
    user=Depends(get_current_user),
    token=Security(get_token_with_scopes, scopes=["write:users"]),
    resource=Depends(load_target_user),
):
    ...
```

---

## Key Takeaways

- **Authentication** answers "who is this user?"; **authorization** answers "what can this user do?" Both are needed.
- **RBAC** is the right starting point: roles like admin, member, viewer; permissions mapped to roles; checks at the action level. Use constants for role names; typos in role strings are silent authorization bypasses.
- **OAuth2 scopes** are the right tool for API tokens. `Security(get_token, scopes=[...])` enforces them in FastAPI.
- **ABAC** handles complex rules like "user can edit their own post only if it's not locked." Use a policy class; don't hardcode the rules in handlers.
- **Row-Level Security** in PostgreSQL is the right tool for "this row belongs to user X" rules. The database enforces the policy; the application never has to filter.
- The combined model: **scopes first** (cheapest, on the token), **roles next** (one DB query), **ABAC last** (requires the resource).
- **Never trust the client** for ownership, role, or any authorization decision. The server sets ownership from the authenticated identity.
- The **Open Policy Agent (OPA)** is the standard for complex, runtime-editable policies. The service asks OPA "is this allowed?" and OPA returns true/false.
- Always test the **deny paths**, not just the allow paths. A test that checks "admin can access" is incomplete without "member cannot access."

## References

- [NIST RBAC Model](https://csrc.nist.gov/projects/role-based-access-control)
- [RFC 6749 — OAuth 2.0](https://www.rfc-editor.org/rfc/rfc6749)
- [FastAPI Security — OAuth2 with Password Hashing](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/)
- [OWASP Authorization Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authorization_Cheat_Sheet.html)
- [Open Policy Agent](https://www.openpolicyagent.org/)
- [Casbin Python](https://casbin.org/docs/en/overview)
- [PostgreSQL Row Security Policies](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
- [Auth0 — Roles and Permissions](https://auth0.com/docs/manage-users/access-control/rbac)
