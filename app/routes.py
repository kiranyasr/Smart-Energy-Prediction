import os
import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, send_file, flash

main = Blueprint("main", __name__)

UPLOAD_FOLDER = "uploads"
CLEANED_FOLDER = "cleaned"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLEANED_FOLDER, exist_ok=True)


@main.route("/")
def home():
    return redirect(url_for("main.upload"))


# ---------- UPLOAD ROUTE ----------
@main.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "dataset" not in request.files:
            flash("No file uploaded", "danger")
            return redirect(request.url)

        file = request.files["dataset"]

        if file.filename == "":
            flash("No selected file", "danger")
            return redirect(request.url)

        if file and file.filename.endswith(".csv"):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Clean dataset (drop NA, reset index)
            df = pd.read_csv(filepath)
            df_cleaned = df.dropna().reset_index(drop=True)

            # Save cleaned file
            cleaned_file = f"cleaned_{file.filename}"
            cleaned_path = os.path.join(CLEANED_FOLDER, cleaned_file)
            df_cleaned.to_csv(cleaned_path, index=False)

            # Preview first 10 rows as HTML table
            preview_html = df_cleaned.head(10).to_html(classes="table table-dark table-striped", index=False)

            return render_template(
                "upload.html",
                preview=preview_html,
                cleaned_file=cleaned_file
            )

    return render_template("upload.html")


# ---------- DOWNLOAD CLEANED FILE ----------
@main.route("/download/<filename>")
def download_cleaned(filename):
    filepath = os.path.join(CLEANED_FOLDER, filename)
    return send_file(filepath, as_attachment=True)


# ---------- DASHBOARD ----------
@main.route("/dashboard")
def dashboard():
    # Load last cleaned dataset if exists
    try:
        files = os.listdir(CLEANED_FOLDER)
        if not files:
            flash("No dataset uploaded yet!", "warning")
            return redirect(url_for("main.upload"))

        latest_file = sorted(files)[-1]
        df = pd.read_csv(os.path.join(CLEANED_FOLDER, latest_file))

        # Assume solar & wind columns exist
        solar = df["Solar"].tolist() if "Solar" in df.columns else []
        wind = df["Wind"].tolist() if "Wind" in df.columns else []
        combined = [(s + w) / 2 for s, w in zip(solar, wind)] if solar and wind else []

        return render_template(
            "dashboard.html",
            solar_data=solar,
            wind_data=wind,
            combined_data=combined
        )
    except Exception as e:
        flash(f"Error loading dashboard: {str(e)}", "danger")
        return redirect(url_for("main.upload"))
