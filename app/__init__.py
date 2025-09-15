from flask import Flask
import os

def create_app():
    # Set template and static folder explicitly
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
        static_folder=os.path.join(os.path.dirname(__file__), '..', 'static')
    )
    app.secret_key = "secret-key"

    from .routes import main
    app.register_blueprint(main)

    return app
