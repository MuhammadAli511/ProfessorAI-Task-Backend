from flask import Flask
from database import chat_db
from routes.documents import documents_route
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.register_blueprint(documents_route)


@app.route("/")
def hello():
    return "<h1>Hello, World!</h1>"


if __name__ == "__main__":
    app.run()
