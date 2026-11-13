"""
auth.py
-------
A lightweight self-service registration + login system, capped at
3 total user accounts.

No external database is used - user accounts are stored in a small local
JSON file (users.json), created automatically the first time someone
registers. This keeps things simple enough for a learning project while
still giving each person their own private login.

How it works:
    1. Anyone can register their own username/password, as long as fewer
       than MAX_USERS accounts exist yet.
    2. Passwords are stored as SHA-256 hashes, never in plain text.
    3. Once logged in, Streamlit's st.session_state keeps that person's
       session separate from everyone else's (each browser = its own
       session), so multiple registered users can use the app at the
       same time without interfering with each other.

⚠️ Security note (important for beginners):
    SHA-256 hashing (without salting) and a plain JSON file are fine for a
    learning project like this, but this is NOT production-grade security.
    A real app would use salted hashing (bcrypt/argon2), a proper database,
    HTTPS, and rate limiting on login attempts.
"""

import hashlib
import json
import os
import streamlit as st

from styles import load_css

USERS_FILE = "users.json"
MAX_USERS = 3  # only up to 3 people can register


def hash_password(password: str) -> str:
    """Turn a plain-text password into a SHA-256 hash string."""
    return hashlib.sha256(password.encode()).hexdigest()


def load_users() -> dict:
    """Load all registered users from users.json (empty dict if none yet)."""
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_users(users: dict) -> None:
    """Save the users dict back to users.json."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def register_user(username: str, password: str, confirm_password: str):
    """
    Try to register a new user.
    Returns (success: bool, message: str).
    """
    username = username.strip()

    if not username or not password:
        return False, "Username and password are required."

    if password != confirm_password:
        return False, "Passwords do not match."

    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    users = load_users()

    if len(users) >= MAX_USERS:
        return False, f"Registration is closed - the maximum of {MAX_USERS} users has been reached."

    if username in users:
        return False, "That username is already taken."

    users[username] = {"password_hash": hash_password(password)}
    save_users(users)

    return True, "Account created! You can now log in."


def check_login(username: str, password: str) -> bool:
    """Return True if the username exists and the password hash matches."""
    users = load_users()
    if username in users:
        return users[username]["password_hash"] == hash_password(password)
    return False


def _inject_login_css():
    load_css("login_style.css")


def login_screen():
    """
    Render a centered Login / Register screen (as two tabs).
    Stops the rest of the app from running until the user logs in.
    """
    _inject_login_css()

    _, center_col, _ = st.columns([1, 1.2, 1])

    with center_col:
        st.markdown(
            """
            <div class="login-wrapper">
                <div class="login-title">💹 Smart Wallet Analyzer</div>
                <div class="login-subtitle">Sign in or create your own account</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs(["🔑 Log In", "📝 Register"])

        # --- LOG IN TAB ---
        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                if check_login(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

        # --- REGISTER TAB ---
        with tab_register:
            users_count = len(load_users())
            slots_left = MAX_USERS - users_count

            if slots_left <= 0:
                st.warning(f"Registration is closed - the maximum of {MAX_USERS} users has been reached.")
            else:
                st.caption(f"{slots_left} of {MAX_USERS} account slot(s) remaining.")
                with st.form("register_form"):
                    new_username = st.text_input("Choose a username", key="reg_username")
                    new_password = st.text_input("Choose a password", type="password", key="reg_password")
                    confirm_password = st.text_input("Confirm password", type="password", key="reg_confirm")
                    register_submitted = st.form_submit_button("Create Account", use_container_width=True)

                if register_submitted:
                    success, message = register_user(new_username, new_password, confirm_password)
                    if success:
                        st.success(message + " Switch to the 'Log In' tab to sign in.")
                    else:
                        st.error(message)


def require_login():
    """
    Call this at the very top of app.py.
    If the user isn't logged in yet, shows the login/register screen and
    halts the rest of the script (st.stop()) so nothing else renders.
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        login_screen()
        st.stop()


def logout_button():
    """Render a logout button (call this from the sidebar)."""
    if st.session_state.get("authenticated"):
        st.sidebar.markdown(f"👤 Logged in as **{st.session_state.username}**")
        if st.sidebar.button("Log Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
