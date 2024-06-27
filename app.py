from flask import Flask, request, jsonify
from swagger_gen.lib.wrappers import swagger_metadata
from swagger_gen.swagger import Swagger
from main import CrewRunner

app = Flask(__name__)


@app.route("/classify_review", methods=["POST"])
@swagger_metadata(
    request_model={"review": "string"},
    response_model=[(200, "Success"), (500, "Error")],
    summary="Submit review for analisys",
)
def classify_review():
    review = request.get_json()["review"]

    result = CrewRunner().run_crew(review)

    return result


swagger = Swagger(app=app, title="app")
swagger.configure()

if __name__ == "__main__":
    app.run(debug=True)
