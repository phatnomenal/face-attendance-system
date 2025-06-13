from flask import Flask, render_template, request, redirect, flash, send_file
import subprocess
import pandas as pd
from datetime import datetime, timedelta
import os
from utils import erase_person, fix_name, get_all_labels, generate_report_csv


app = Flask(__name__)
app.secret_key = 'fatz'  # Required for flashing messages

@app.route('/')
def index():
    people = get_all_labels()
    return render_template('index.html', people=people)

@app.route('/add', methods=['POST'])
def add_person():
    name = request.form.get('name')
    if name:
        subprocess.run(['python', 'test_faces_full_workflow.py', '--mode', 'add', '--name', name])
    return redirect('/')


@app.route('/erase', methods=['POST'])
def erase():
    name = request.form.get('name')
    if name:
        success = erase_person(name)
        if success:
            flash(f"Erased {name}")
        else:
            flash(f"{name} not found.")
    return redirect('/')

@app.route('/fix', methods=['POST'])
def fix():
    old = request.form.get('old_name')
    new = request.form.get('new_name')
    if old and new:
        updated = fix_name(old, new)
        if updated:
            flash(f"Updated '{old}' to '{new}'")
        else:
            flash(f"Name '{old}' not found.")
    return redirect('/')


@app.route('/attendance')
def attendance():
    subprocess.run(['python', 'test_faces_full_workflow.py', '--mode', 'recognize'])
    return redirect('/')

@app.route('/report/<period>')
def report(period):
    filename = generate_report_csv(period, session_gap_minutes=10)
    if filename:
        return send_file(filename, as_attachment=True)
    flash("Failed to generate report.")
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)
