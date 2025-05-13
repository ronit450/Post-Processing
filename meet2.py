# Week1 
# asking user to enter their name
usr_name = input("Please Enter your name: ")
print(f"Hello, {usr_name} !")


# Week 2
# asking user to enter their age
usr_age = int(input("Please Enter your age: "))
# as age is integer so if it is greater than 18, it means he is adult otherwise he is minor
if usr_age > 18:
    print(f"You are {usr_age} years old and you are an adult.")
elif usr_age < 0:
    print("Invalid age, it should be greater than 0")
else:
    print(f"You are {usr_age} years old. You are a minor ")


# week 3:
# defining sample list for the user and adding them to the list
temp_lst = [5,4,3,2,1]
# for the first three numbers, we will only access 0, 1, 2 index as python starts indexing from 0
temp_sum = temp_lst[0] + temp_lst[1] + temp_lst[2]
print(temp_sum) # printing the sum


# week 4
# Asking the user to enter the table for multiplication
table_num = int(input("Enter the number you want the table for"))
# accessing the loop from 1 to 11 as python runs till n-1
for i in range(1,11):
    print(f"{table_num} x {i} = {table_num*i}") # table number * i would be for the display and i would be the index


# week 5
# defining the function which will take the list and with loop checking the max number
def max_checker(temp_lst):
    # checking if the user has not entered the empty list as acessing the list would give error
    if temp_lst == []:
        return "Empty List"
    # by default we assume the first index is the max number 
    max_num = temp_lst[0]
    
    # assessing the other elements in the list to check
    for i in temp_lst:
        if i > max_num:
            # if we found any number greater than max_num, we will replace it
            max_num = i
    return max_num

max_num = max_checker(temp_lst)
print(max_num)


# Week 6 
# asking user to enter the sentence
sentence = input("Please Enter your sentence and I will check number of vowels : ")

# definign the basic vowels 
temp_vowels = ['a', 'e', 'i', 'o', 'u']
count = 0
for i in sentence:
    if i.lower() in temp_vowels: # to save from checking the capital letters we will lower the string value 
        count += 1

# presenting the count 
print("Number of vowels in your sentence is: ", count)