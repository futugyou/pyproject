from flask import Flask, request, jsonify
from client import AuthClient
from config import AuthOptions


def main():
    opts = AuthOptions(
        auth_server_url="https://auth.example.com",
        client_id="client-id",
        client_secret="secret",
        scopes=["openid", "profile"],
        redirect_url="http://localhost:5000/oauth/callback",
        auth_url="/authorize",
        token_url="/token",
        db_url="mongodb://localhost:27017",
        db_name="oauth_db",
    )

    auth_service = AuthClient(opts)

    app = Flask(__name__)
    
    @app.route("/oauth/start")
    def start_oauth():
        url = auth_service.create_auth_code_url()
        return redirect(url)

    @app.route("/oauth/callback")
    def oauth_callback():
        code = request.args.get("code")
        state = request.args.get("state")
        token = auth_service.exchange_token(code, state)
        return jsonify(token)

    @app.route("/oauth/verify")
    def verify():
        token = request.args.get("token")
        result = auth_service.verify_token(token)
        return jsonify(result)


if __name__ == "__main__":
    main()
