from flask import Flask, jsonify
from models.recipe import get_ingredients_to_expire, get_recipe
from models.defaultroberta import update_expirations_in_firestore

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

@app.route("/update_expirations", methods=["POST"])  # Use POST because this endpoint modifies data
def update_expirations_endpoint():
    # This endpoint will call the function to update the "daysTillExpire" for each item
    try:
        update_expirations_in_firestore()
        return jsonify({"status": "success", "message": "Expirations updated successfully!"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000)
