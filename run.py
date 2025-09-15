from app import create_app

# Create the Flask app
app = create_app()

if __name__ == "__main__":
    # Run on http://127.0.0.1:5000
    app.run(debug=True)
