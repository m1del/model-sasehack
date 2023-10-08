from flask import Flask, jsonify
from models.recipe import get_ingredients_to_expire, get_recipe

app = Flask(__name__)

@app.route("/expiring_ingredients", methods=["GET"])
def expiring_ingredients_endpoint():
    ingredients, _ = get_ingredients_to_expire()
    return jsonify({"expiring_ingredients": ingredients})

@app.route("/get_recipe", methods=["GET"])
def get_recipe_endpoint():
    ingredients, inventory = get_ingredients_to_expire()
    recipe = get_recipe(ingredients, inventory)
    return jsonify({"recipe": recipe})

if __name__ == "__main__":
    app.run(port=5000)
