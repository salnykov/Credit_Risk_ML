from data_cleanup import * 
import matplotlib.pyplot as plt

# Display descriptive statistics of raw data
print(descriptive(data))
print('')
print('-----------------')
print('')

# Study scatter of person_age and person_emp_length because it may contain 
# identifiable outliers
plt.scatter(data['person_age'], data['person_emp_length'], c='blue', alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Person Employment Length")

# It is highly unlikely that a borrower has work experience above 60 years or 
# age above 100 years. Hence, observations with values above these will be 
# considered outliers
plt.axvline(x=100, color='red')
plt.axhline(y=60, color='red')
plt.show()

print ('Dataset free from obvious outliers')
print('')
print (descriptive(data_clean0))
print('')
print('-----------------')
print('')

print ('Dataset free from obvious outliers and missing values')
print('')
print (descriptive(data_clean))
print('')
print('-----------------')
print('')

