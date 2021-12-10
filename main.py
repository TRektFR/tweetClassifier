import joblib
from flask import Flask, request, render_template

model = joblib.load("model.pkl")

classes= {0: "hate_speech", 1: "offensive_language", 2: "neither"}

app = Flask(__name__)


@app.route("/tweetclf", methods=["GET", "POST"])
def home():
    tweet_type = -1
    text = ""
    if request.method == 'POST':
        text = request.form.get('txt')
        tweet_type = classes[get_class(text)]
    return render_template("main.html",txt = text , type=tweet_type)


def get_class(sentence):
    pred= model.predict([sentence])
    print(pred)
    return pred[0]


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
