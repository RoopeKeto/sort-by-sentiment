from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField
from wtforms.validators import DataRequired, Length, NumberRange

class SearchForm(FlaskForm):
    searchword = StringField('Search word',
                             validators=[DataRequired(), Length(min=1, max=200)])
    number_of_results = IntegerField('Number of tweets', 
                                    validators=[DataRequired(), NumberRange(min=1, max=1500) ])
    search = SubmitField('Search')