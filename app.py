from flask import Flask, jsonify, request
from flask_cors import CORS
from Router import router_excutoer
app = Flask(__name__)
CORS(app)

@app.route('/API/aboutDeveloper', methods=['POST'])
def about_developer():
    data = request.get_json()
    input = data['input']
    res = router_excutoer(input)
    return jsonify({"response" : res})


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=8080)