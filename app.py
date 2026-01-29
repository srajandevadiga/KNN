import joblib
from flask import Flask, render_template, request
from sklearn.neighbors import KNeighborsClassifier
app = Flask(__name__)
x = [[20], [30], [40], [50], [55], [65], [75]]
y = [0, 0, 0, 1, 1, 1, 1]   # labels (example: young=0, old=1)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, y)
joblib.dump(knn, 'model.pkl')
@app.route("/", methods=["GET", "POST"])
def knn_predict():
    prediction = ""
    if request.method == "POST":
        age = int(request.form["age"])
        model = joblib.load("model.pkl")
        result = model.predict([[age]])
        prediction = "OLD" if result[0] == 1 else "YOUNG"

    return render_template("index.html", prediction=prediction)
if __name__ == "__main__":
    app.run(debug=True)
