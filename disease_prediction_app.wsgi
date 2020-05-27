#flaskapp.wsgi
import sys
sys.path.insert(0, '/var/www/html/disease_prediction_app')

from diseasePredict import app as application