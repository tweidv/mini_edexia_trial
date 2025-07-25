from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from diagram_hidden.diagram_workflow import run_diagram_workflow_for_web

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Diagram workflow tool
@app.route('/diagram-workflow', methods=['GET', 'POST'])
def diagram_workflow():
    if request.method == 'POST':
        file = request.files.get('image')
        question = request.form.get('question')
        if not file or not question:
            return render_template('diagram_workflow.html', error='Please provide both an image and a question.')
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result = run_diagram_workflow_for_web(filepath, question)
        return render_template('diagram_workflow.html', result=result, image_url=url_for('static', filename=filename))
    return render_template('diagram_workflow.html')

if __name__ == '__main__':
    app.run(debug=True) 