from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo
from wtforms import widgets, SelectMultipleField


class MultiCheckboxField(SelectMultipleField):
    widget = widgets.ListWidget(prefix_label=False)
    option_widget = widgets.RadioInput()


class RegistrationForm(FlaskForm):
    list_of_triple_options: list = ['Normal', 'Above Normal', 'Very High']
    # create a list of value/description tuples
    triple_options = [(x, x) for x in list_of_triple_options]

    list_of_double_options: list = ['Yes', 'No']
    # create a list of value/description tuples
    double_options = [(x, x) for x in list_of_double_options]

    list_of_gender_options: list = ['Male', 'Female']
    # create a list of value/description tuples
    gender_options = [(x, x) for x in list_of_gender_options]

    patient_name = StringField('Full Name', validators=[DataRequired()])
    age = StringField('Age in years', validators=[DataRequired()])
    height = StringField('Height in cm', validators=[DataRequired()])
    weight = StringField('Weight in kg', validators=[DataRequired()])
    s_bp = StringField('Systolic Blood Pressure in mm Hg', validators=[DataRequired()])
    d_bp = StringField('Diastolic Blood Pressure in mm Hg', validators=[DataRequired()])

    gh_level = MultiCheckboxField('Glucose Level', choices=triple_options)
    ch_level = MultiCheckboxField('Cholesterol Level', choices=triple_options)

    smoke = MultiCheckboxField('Do you smoke?', choices=double_options)
    alcohol = MultiCheckboxField('Do you drink?', choices=double_options)
    activity = MultiCheckboxField('Do you regularly exercise?', choices=double_options)

    gender = MultiCheckboxField('Gender', choices=gender_options)

    test = SubmitField('Test')
