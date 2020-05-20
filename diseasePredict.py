from flask import Flask, render_template, flash, redirect, url_for, request
import os
from forms import RegistrationForm
import disease

app = Flask(__name__)

app.config['SECRET_KEY'] = 'd5b22a2f57aa2a933a99be291d616654'
prediction_data: dict = {}


@app.route("/", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        prediction_data['age_yr'] = int(form.age.data)
        prediction_data['height'] = int(form.height.data)
        prediction_data['weight'] = int(form.weight.data)
        prediction_data['gender'] = form.gender.data
        prediction_data['ap_hi'] = int(form.s_bp.data)
        prediction_data['ap_lo'] = int(form.d_bp.data)
        prediction_data['cholesterol'] = form.ch_level.data
        prediction_data['gluc'] = form.gh_level.data
        prediction_data['smoke'] = form.smoke.data
        prediction_data['alco'] = form.alcohol.data
        prediction_data['active'] = form.activity.data
        return redirect(url_for('result'))

    return render_template('register.html', title='Details', form=form)


@app.route("/result", methods=['GET'])
def result():
    form = RegistrationForm()
    final_prediction_json: dict = processed_data(prediction_data)
    print(final_prediction_json)
    prediction_value = disease.predict_on_user_input(final_prediction_json)
    print("final prediction value is " + str(prediction_value))

    if prediction_value == 0:
        result_to_display: str = "Our Diagnosis suggests that patient does not suffers from any cardiovascular disease."
    else:
        result_to_display: str = "Our diagnosis suggests patient does suffer from cardiovascular disease.\nPlease get checked soon."

    return render_template('prediction.html', title='Result', result_to_display=result_to_display)


def processed_data(input_dict: dict):
    for dict_key in input_dict.keys():
        if dict_key in ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'gender']:
            if input_dict[dict_key][0] == "Normal":
                input_dict[dict_key] = 1
            elif input_dict[dict_key][0] == "Above Normal":
                input_dict[dict_key] = 2
            elif input_dict[dict_key][0] == "Very High":
                input_dict[dict_key] = 3
            elif input_dict[dict_key][0] == "Yes":
                input_dict[dict_key] = 1
            elif input_dict[dict_key][0] == "No":
                input_dict[dict_key] = 0
            elif input_dict[dict_key][0] == "Male":
                input_dict[dict_key] = 1
            elif input_dict[dict_key][0] == "Female":
                input_dict[dict_key] = 2

    return input_dict
