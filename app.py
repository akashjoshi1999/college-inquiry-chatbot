from flask import Flask, render_template, request
import app_test

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/home")
def homm2():
    return render_template("home.html")

@app.route("/about")
def homm3():
    return render_template("about.html")

@app.route("/contact")
def home4():
    return render_template("contact.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return app_test.chatResp(userText)
if __name__ == "__main__":
    app.run()