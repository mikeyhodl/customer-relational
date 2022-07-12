from urllib import response
from flask import Flask, render_template, request
from flask import make_response
import warnings
warnings.filterwarnings('ignore')
import os

from bot.chatbot import get_response

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def test():
    return "SHADOW BOT"

@app.route("/api", methods=['GET', 'POST'])
def api():
    # user_input = request.json['user_input']
    user_input = request.args.get('user_input')
    # if request.method == 'POST':
    # user_input = request.form['user_input']

    # user_input = "Hello"

    bot_response = get_response(user_input)
    response = make_response({"response": bot_response})
    # response.headers['Content-Type'] = 'application/json'
    return response

if __name__ == '__main__':

    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8085)))
