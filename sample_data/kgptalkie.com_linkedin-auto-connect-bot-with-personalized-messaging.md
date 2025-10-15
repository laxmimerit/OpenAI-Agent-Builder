https://kgptalkie.com/linkedin-auto-connect-bot-with-personalized-messaging

# LinkedIn Auto Connect Bot with Personalized Messaging

Published by  
georgiannacambel  
on  
5 September 2020

## Auto Connect Bot for LinkedIn

In this project, we are going to create a bot that finds the people in your LinkedIn suggestions and sends a connection request to each one of them with a message. It also finds the suggestions of your suggestions and sends them a connection request.

### Required Packages

You will need to install the following packages:

- **selenium**
- **beautifulsoup4**

Below are the commands to install them:

```bash
!pip install selenium
```

[https://pypi.org/project/selenium/](https://pypi.org/project/selenium/)  
[https://selenium-python.readthedocs.io/api.html](https://selenium-python.readthedocs.io/api.html)

```bash
!pip install beautifulsoup4
```

[https://pypi.org/project/beautifulsoup4/](https://pypi.org/project/beautifulsoup4/)

### Web Driver

Besides these packages, you also need to download a **web driver** for your browser. Based on your Google Chrome version, you can download the driver from [here](https://chromedriver.chromium.org/). Save it in the working repository.

### Configuration File

Lastly, in the `config.txt` file, you need to add your email ID and LinkedIn password. Save that file in the working repository.

---

## Code Implementation

### Importing Libraries

Here we have imported the necessary libraries:

```python
import os, random, sys, time
from selenium import webdriver
from bs4 import BeautifulSoup
```

### Initializing the Browser

Here we are getting the address of the Google Chrome driver. After you run this line, a new Google Chrome window will open:

```python
browser = webdriver.Chrome('driver/chromedriver.exe')
```

### Logging into LinkedIn

Now we will open the LinkedIn login page:

```python
browser.get('https://www.linkedin.com/uas/login')
```

We will open the `config.txt` file and read the username and password:

```python
file = open('config.txt')
lines = file.readlines()
username = lines[0]
password = lines[1]
```

Automating the login process:

```python
elementID = browser.find_element_by_id('username')
elementID.send_keys(username)

elementID = browser.find_element_by_id('password')
elementID.send_keys(password)

elementID.submit()
```

**Note:** The IDs of the textboxes can change. Hence, before running this code, check the current ID of the textboxes by inspecting the webpage.

---

### Visiting Profiles

Now we will create links to visit different profiles:

```python
visitingProfileID = '/in/aarya-tadvalkar-092650193/'
fullLink = 'https://www.linkedin.com' + visitingProfileID
browser.get(fullLink)
```

---

## Extracting Profile IDs

We will write a function to return the list of profile IDs suggested by LinkedIn:

```python
visitedProfiles = []
profilesQueued = []

def getNewProfileIDs(soup, profilesQueued):
    profilesID = []
    pav = soup.find('div', {'class': 'pv-browsemap-section'})
    all_links = pav.findAll('a', {'class': 'pv-browsemap-section__member'})
    for link in all_links:
        userID = link.get('href')
        if ((userID not in profilesQueued) and (userID not in visitedProfiles)):
            profilesID.append(userID)
    return profilesID
```

**Note:** The class names used here can change. Hence, before running this code, check the current class name by inspecting the webpage.

---

## Sending Connection Requests

Now for each link in `profilesQueued`, we will perform the following actions:

1. Append the Profile ID to the base link `https://www.linkedin.com` to get the full link.
2. Visit the full link.
3. Send a connection request with a personalized message.

### Code Implementation

```python
while profilesQueued:
    try:
        visitingProfileID = profilesQueued.pop()
        visitedProfiles.append(visitingProfileID)
        fullLink = 'https://www.linkedin.com' + visitingProfileID
        browser.get(fullLink)

        # Click the connect button
        browser.find_element_by_class_name('pv-s-profile-actions').click()

        # Click the "Add a note" button
        browser.find_element_by_class_name('mr1').click()

        # Send a personalized message
        customMessage = "Hello, I have found mutual interest area and I would be more than happy to connect with you. Kindly, accept my invitation. Thanks!"
        elementID = browser.find_element_by_id('custom-message')
        elementID.send_keys(customMessage)

        # Click the "Done" button
        browser.find_element_by_class_name('ml1').click()

        # Add the ID to the visitedUsers.txt file
        with open('visitedUsers.txt', 'a') as visitedUsersFile:
            visitedUsersFile.write(str(visitingProfileID) + '\n')
        visitedUsersFile.close()

        # Get new profiles ID
        soup = BeautifulSoup(browser.page_source)
        try:
            profilesQueued.extend(getNewProfileIDs(soup, profilesQueued))
        except:
            print('Continue')

        # Random delay to avoid detection
        time.sleep(random.uniform(3, 7))

        # Print progress
        if (len(visitedProfiles) % 50 == 0):
            print('Visited Profiles: ', len(visitedProfiles))

        if (len(profilesQueued) > 100000):
            with open('profilesQueued.txt', 'a') as visitedUsersFile:
                visitedUsersFile.write(str(visitingProfileID) + '\n')
            visitedUsersFile.close()
            print('100,000 Done!!!')
            break
    except:
        print('error')
```

---

**Source:** https://kgptalkie.com/linkedin-auto-connect-bot-with-personalized-messaging