import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Gender':1, 'Python_exp':54400, 'Experience_Years':0, 
                            'Education':8, 'Internship':4, 'Score':23625, 'Salary * 10E4':14.27, 
                            'Offer_History':0.44, 'Location':0, 'Recruitment_Status':4})

print(r.json())