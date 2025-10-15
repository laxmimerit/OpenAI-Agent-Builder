https://kgptalkie.com/linkedin-profile-scrapper-in-python

# LinkedIn Profile Scrapper in Python

Published by [georgiannacambel](https://kgptalkie.com/linkedin-profile-scrapper-in-python) on 6 September 2020

## LinkedIn Profile Scraping using Selenium and Beautiful Soup

Scraping of LinkedIn profiles is a very useful activity especially to achieve public relations/marketing tasks. In this project, we are going to scrap important data from a LinkedIn profile.

### Installation

The first part of this project is to automatically log in to our LinkedIn account. For that, you will have to download a *web driver*. You will also have to install [Selenium](https://pypi.org/project/selenium/) and [beautifulsoup4](https://pypi.org/project/beautifulsoup4/). Below are the commands which you need to run to download both the packages. You can also visit the given links for more information about installation.

For Selenium:-
```python
!pip install selenium
```
https://pypi.org/project/selenium/
https://selenium-python.readthedocs.io/api.html

For beautifulsoup4:-
```python
!pip install beautifulsoup4
```
https://pypi.org/project/beautifulsoup4/

Based on your Google Chrome version you can download the web driver from [here](https://kgptalkie.com/linkedin-profile-scrapper-in-python). Save it in the working repository.

Lastly, in the `config.txt` file you need to add your email id and LinkedIn password and save that file in the working repository.

For more details related to this you can watch this video. You can even refer to this blog which gives a detailed explanation of the code.

## LinkedIn Auto Connect Bot with Personalized Messaging

Here we have imported the necessary libraries.

```python
import requests, time, random
from bs4 import BeautifulSoup
from selenium import webdriver
```

Here we are getting the address of the Google Chrome driver using `browser = webdriver.Chrome('driver/chromedriver.exe')`. Then we will open the LinkedIn login page using `browser.get()`. We will open the `config.txt` file which we have created and read the `username` and `password` from the file.

Now we have to automate the login process. For that, we will have to check the `id` of the textboxes which accept the username and password on the webpage. We can do this by right-clicking anywhere on the webpage and then clicking on ‘inspect’. After doing this you will see that the `id` of the username textbox is `username` and the `id` of password textbox is `password`.

`find_element_by_id()` returns the first element with the id attribute value matching the location. `send_keys()` method is used to send text to any field, such as input field of a form or even to anchor tag paragraph, etc. It replaces its contents on the webpage in your browser. `submit()` method is used to submit a form after you have sent data to a form.

```python
browser = webdriver.Chrome('driver/chromedriver.exe')
browser.get('https://www.linkedin.com/uas/login')
file = open('config.txt')
lines = file.readlines()
username = lines[0]
password = lines[1]
```

**Source:** https://kgptalkie.com/linkedin-profile-scrapper-in-python

```python
elementID = browser.find_element_by_id('username')
elementID.send_keys(username)
```

**Source:** https://kgptalkie.com/linkedin-profile-scrapper-in-python

```python
elementID = browser.find_element_by_id('password')
elementID.send_keys(password)
```

**Source:** https://kgptalkie.com/linkedin-profile-scrapper-in-python

```python
elementID.submit()
```

`link` contains the link of the profile we want to scrap. You can scrap any profile of your choice or you can even scrap multiple links using a `for` loop.

```python
link = 'https://www.linkedin.com/in/rishabh-singh-61b706114/'
browser.get(link)
```

Watch Video for this blog:

The whole profile doesn’t get loaded at the start. Only the part which we can see is loaded. So we will have to scroll the profile till the end so that the complete profile is loaded. The code given below scrolls the profile till the end.

```python
SCROLL_PAUSE_TIME = 5
# Get scroll height
last_height = browser.execute_script("return document.body.scrollHeight")
for i in range(3):
    # Scroll down to bottom
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)
    # Calculate new scroll height and compare with last scroll height
    new_height = browser.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height
```

Now as the full page is loaded, you are ready to get the page source. We will use the `lxml` parser and the source code in a `BeautifulSoup` object `soup`.

```python
src = browser.page_source
soup = BeautifulSoup(src, 'lxml')
```

To extract anything from the webpage we will have to inspect the webpage. We can do this by right-clicking anywhere on the webpage and then clicking on ‘inspect’.

The block containing the basic information is represented using the `div` tag with class name as `flex-1 mr5`.

```python
name_div = soup.find('div', {'class': 'flex-1 mr5'})
```

```html
<div class="flex-1 mr5">
  <ul class="pv-top-card--list inline-flex align-items-center">
    <li class="inline t-24 t-black t-normal break-words">
      Rishabh Singh
    </li>
    <li class="pv-top-card__distance-badge inline-block v-align-text-bottom t-16 t-black--light t-normal"><span class="distance-badge separator">
      <span class="visually-hidden">3rd degree connection</span><span aria-hidden="true" class="dist-value">3rd</span>
    </span></li>
    <!-- --> <li class="inline-flex ml2">
      <span class="pv-member-badge--for-top-card inline-flex pv-member-badge ember-view" id="ember102" style="display: none;"><!-- -->
        <!-- -->
        <span class="visually-hidden">
          Rishabh has a  account
        </span>
        <!-- --></span>
    </li>
    <!-- --> </ul>
  <h2 class="mt1 t-18 t-black t-normal break-words">
    #futureshaper
  </h2>
  <ul class="pv-top-card--list pv-top-card--list-bullet mt1">
    <li class="t-16 t-black t-normal inline-block">
      Bengaluru, Karnataka, India
    </li>
    <!-- -->
    <li class="inline-block">
      <span class="t-16 t-black t-normal">
        500+ connections
      </span>
    </li>
    <li class="inline-block">
      <a class="ember-view" data-control-name="contact_see_more" href="/in/rishabh-singh-61b706114/detail/contact-info/" id="ember103"> <span class="t-16 link-without-visited-state">
        Contact info
      </span>
      </a> </li>
  </ul>
</div>
```

We will first get the name. As you can see `name_div` there are 2 `ul` tags. The first `ul` consists of the name and the second `ul` consists of the location and no. of connections.

Here we will first get both the `ul` tags using `name_div.find_all('ul')`. We will find the `li` in the first `ul` tag using `name_loc[0].find('li')` and get the text enclosed in it using `get_text()`.

```python
name_loc = name_div.find_all('ul')
name = name_loc[0].find('li').get_text().strip()
name
'Rishabh Singh'
```

Similarly, for the location we will find the `li` in the second `ul` tag.

```python
loc = name_loc[1].find('li').get_text().strip()
loc
'Bengaluru, Karnataka, India'
```

The profile title is enclosed in the `h2` tag. So we can extract it using `name_div.find('h2').get_text()`.

```python
profile_title = name_div.find('h2').get_text().strip()
profile_title
'#futureshaper'
```

The no. of connections is in 2nd `li` of the 2nd `ul`. Hence first we will find all the `li` tags in the second `ul` using `name_loc[1].find_all('li')`. Then we will get the text from the second `li` tag using `connection[1].get_text()`.

```python
connection = name_loc[1].find_all('li')
connection = connection[1].get_text().strip()
connection
'500+ connections'
```

We will append everything we have scrapped till now in `info`.

```python
info = []
info.append(link)
info.append(name)
info.append(profile_title)
info.append(loc)
info.append(connection)
info
['https://www.linkedin.com/in/rishabh-singh-61b706114/',
 'Rishabh Singh',
 '#futureshaper',
 'Bengaluru, Karnataka, India',
 '500+ connections']
```

### Experience

Now we will scrap the information under the experience section in the profile. We can access the experience section using the tag `section` and id `experience-section`.

```python
exp_section = soup.find('section', {'id': 'experience-section'})
```

```html
<section class="pv-profile-section experience-section ember-view" id="experience-section"><header class="pv-profile-section__card-header">
  <h2 class="pv-profile-section__card-heading">
    Experience
  </h2>
  <!-- --></header>
  <ul class="pv-profile-section__section-info section-info pv-profile-section__section-info--has-no-more">
    <li class="pv-entity__position-group-pager pv-profile-section__list-item ember-view" id="ember166"> <section class="pv-profile-section__card-item-v2 pv-profile-section pv-position-entity ember-view" id="1517647779"> <div class="display-flex justify-space-between full-width">
      <div class="display-flex flex-column full-width">
        <a class="full-width ember-view" data-control-name="background_details_company" href="/company/honeywell/" id="ember168"> <div class="pv-entity__logo company-logo">
          <img alt="Honeywell" class="pv-entity__logo-img EntityPhoto-square-5 lazy-image ember-view" id="ember170" loading="lazy" src="https://media-exp1.licdn.com/dms/image/C560BAQFvcIh3UnA5zw/company-logo_100_100/0?e=1607558400&amp;v=beta&amp;t=dDiL6fU4CW1y7u-RdPOENVsnHqExUQVv9qs_lj14xBw"/>
        </div>
        <div class="pv-entity__summary-info pv-entity__summary-info--background-section">
          <h3 class="t-16 t-black t-bold">FPGA Engineer</h3>
          <p class="visually-hidden">Company Name</p>
          <p class="pv-entity__secondary-title t-14 t-black t-normal">
            Honeywell
          </p>
          <div class="display-flex">
            <h4 class="pv-entity__date-range t-14 t-black--light t-normal">
              <span class="visually-hidden">Dates Employed</span>
              <span>Aug 2019 – Present</span>
            </h4>
            <h4 class="t-14 t-black--light t-normal">
              <span class="visually-hidden">Employment Duration</span>
              <span class="pv-entity__bullet-item-v2">1 yr 2 mos</span>
            </h4>
          </div>
          <h4 class="pv-entity__location t-14 t-black--light t-normal block">
            <span class="visually-hidden">Location</span>
            <span>Bengaluru Area, India</span>
          </h4>
          <!-- -->
        </div>
      </a>
      <!-- --> </div>
      <!-- --> </div>
    </section>
    </li><li class="pv-entity__position-group-pager pv-profile-section__list-item ember-view" id="ember173"> <section class="pv-profile-section__card-item-v2 pv-profile-section pv-position-entity ember-view" id="929137672"> <div class="display-flex justify-space-between full-width">
      <div class="display-flex flex-column full-width">
        <a class="full-width ember-view" data-control-name="background_details_company" href="/company/l&amp;t-technology-services-limited/" id="ember175"> <div class="pv-entity__logo company-logo">
          <img alt="L&amp;T Technology Services Limited" class="pv-entity__logo-img EntityPhoto-square-5 lazy-image ember-view" id="ember177" loading="lazy" src="https://media-exp1.licdn.com/dms/image/C510BAQFFdHnl8nr2KA/company-logo_100_100/0?e=1607558400&amp;v=beta&amp;t=5GTSsopRboozxuQ6y1Y_LDixQebeP2KBYi39Z3jpOdA"/>
        </div>
        <div class="pv-entity__summary-info pv-entity__summary-info--background-section">
          <h3 class="t-16 t-black t-bold">FPGA Design Engineer</h3>
          <p class="visually-hidden">Company Name</p>
          <p class="pv-entity__secondary-title t-14 t-black t-normal">
            L&amp;T Technology Services Limited
            <span class="pv-entity__secondary-title separator">Full-time</span>
          </p>
          <div class="display-flex">
            <h4 class="pv-entity__date-range t-14 t-black--light t-normal">
              <span class="visually-hidden">Dates Employed</span>
              <span>Jan 2017 – Jul 2019</span>
            </h4>
            <h4 class="t-14 t-black--light t-normal">
              <span class="visually-hidden">Employment Duration</span>
              <span class="pv-entity__bullet-item-v2">2 yrs 7 mos</span>
            </h4>
          </div>
          <h4 class="pv-entity__location t-14 t-black--light t-normal block">
            <span class="visually-hidden">Location</span>
            <span>Bengaluru Area, India</span>
          </h4>
          <!-- -->
        </div>
      </a>
      <!-- --> </div>
      <!-- --> </div>
    </section>
    </li> </ul>
    <!-- --></section>
```

From `exp_section` we are going to get the first `ul` tag. Then from the first `ul` tag we are going to get the first `div` tag. Then from the first `div` tag we are going to get the first `a` tag.

```python
exp_section = exp_section.find('ul')
div_tag = exp_section.find('div')
a_tag = div_tag.find('a')
```

```html
<a class="full-width ember-view" data-control-name="background_details_company" href="/company/honeywell/" id="ember168"> <div class="pv-entity__logo company-logo">
  <img alt="Honeywell" class="pv-entity__logo-img EntityPhoto-square-5 lazy-image ember-view" id="ember170" loading="lazy" src="https://media-exp1.licdn.com/dms/image/C560BAQFvcIh3UnA5zw/company-logo_100_100/0?e=1607558400&amp;v=beta&amp;t=dDiL6fU4CW1y7u-RdPOENVsnHqExUQVv9qs_lj14xBw"/>
</div>
<div class="pv-entity__summary-info pv-entity__summary-info--background-section">
  <h3 class="t-16 t-black t-bold">FPGA Engineer</h3>
  <p class="visually-hidden">Company Name</p>
  <p class="pv-entity__secondary-title t-14 t-black t-normal">
    Honeywell
  </p>
  <div class="display-flex">
    <h4 class="pv-entity__date-range t-14 t-black--light t-normal">
      <span class="visually-hidden">Dates Employed</span>
      <span>Aug 2019 – Present</span>
    </h4>
    <h4 class="t-14 t-black--light t-normal">
      <span class="visually-hidden">Employment Duration</span>
      <span class="pv-entity__bullet-item-v2">1 yr 2 mos</span>
    </h4>
  </div>
  <h4 class="pv-entity__location t-14 t-black--light t-normal block">
    <span class="visually-hidden">Location</span>
    <span>Bengaluru Area, India</span>
  </h4>
  <!-- -->
</div>
</a>
```

We can extract the job title using `h3` tag.

```python
job_title = a_tag.find('h3').get_text().strip()
job_title
'FPGA Engineer'
```

The company name is enclosed by the 2nd `p` tag. Hence we can get it by `a_tag.find_all('p')[1].get_text()`.

```python
company_name = a_tag.find_all('p')[1].get_text().strip()
company_name
'Honeywell'
```

For the joining date we will extract the first `h4` tag using `a_tag.find_all('h4')[0]`. Then we will get the second `span` from the first `h4` using `find_all('span')[1]`.

```python
joining_date = a_tag.find_all('h4')[0].find_all('span')[1].get_text().strip()
joining_date
'Aug 2019 – Present'
```

For the duration we will extract the second `h4` tag using `a_tag.find_all('h4')[1]`. Then we will get the second `span` using `find_all('span')[1]`.

```python
exp = a_tag.find_all('h4')[1].find_all('span')[1].get_text().strip()
exp
'1 yr 2 mos'
```

We will append all the scrapped data to `info`.

```python
info
['https://www.linkedin.com/in/rishabh-singh-61b706114/',
 'Rishabh Singh',
 '#futureshaper',
 'Bengaluru, Karnataka, India',
 '500+ connections']
info.append(company_name)
info.append(job_title)
info.append(joining_date)
info.append(exp)
info
['https://www.linkedin.com/in/rishabh-singh-61b706114/',
 'Rishabh Singh',
 '#futureshaper',
 'Bengaluru, Karnataka, India',
 '500+ connections',
 'Honeywell',
 'FPGA Engineer',
 'Aug 2019 – Present',
 '1 yr 2 mos']
```

### Education

Now we will move to the education section. We can extract it using the `section` tag having id as `education-section`. Then we will get the `ul` tag which contains all the information.

```python
edu_section = soup.find('section', {'id': 'education-section'}).find('ul')
```

```html
<ul class="pv-profile-section__section-info section-info pv-profile-section__section-info--has-no-more">
  <li class="pv-profile-section__list-item pv-education-entity pv-profile-section__card-item ember-view" id="356637700"><div class="display-flex justify-space-between full-width">
    <div class="display-flex flex-column full-width">
      <a class="ember-view" data-control-name="background_details_school" href="/school/223389/?legacySchoolId=223389" id="ember183"> <div class="pv-entity__logo">
        <img alt="Technocrats Institute of Technology (Excellence), Anand Nagar, PB No. 24, Post Piplani, BHEL, Bhopal - 462021" class="pv-entity__logo-img pv-entity__logo-img EntityPhoto-square-4 lazy-image ghost-school ember-view" id="ember185" loading="lazy" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"/>
      </div>
      <div class="pv-entity__summary-info pv-entity__summary-info--background-section">
        <div class="pv-entity__degree-info">
          <h3 class="pv-entity__school-name t-16 t-black t-bold">Technocrats Institute of Technology (Excellence), Anand Nagar, PB No. 24, Post Piplani, BHEL, Bhopal - 462021</h3>
          <p class="pv-entity__secondary-title pv-entity__degree-name t-14 t-black t-normal">
            <span class="visually-hidden">Degree Name</span>
            <span class="pv-entity__comma-item">Bachelor of Engineering (B.E.)</span>
          </p>
          <p class="pv-entity__secondary-title pv-entity__fos t-14 t-black t-normal">
            <span class="visually-hidden">Field Of Study</span>
            <span class="pv-entity__comma-item">Electrical, Electronics and Communications Engineering</span>
          </p>
          <p class="pv-entity__secondary-title pv-entity__grade t-14 t-black t-normal">
            <span class="visually-hidden">Grade</span>
            <span class="pv-entity__comma-item">FIRST</span>
          </p>
        </div>
        <p class="pv-entity__dates t-14 t-black--light t-normal">
          <span class="visually-hidden">Dates attended or expected graduation</span>
          <span>
            <time>2012</time> – <time>2016</time>
          </span>
        </p>
        <!-- --></div>
      </a>
      <!-- --> </div>
      <!-- --></div>
    </li>
    <li class="pv-profile-section__list-item pv-education-entity pv-profile-section__card-item ember-view" id="373985416"><div class="display-flex justify-space-between full-width">
      <div class="display-flex flex-column full-width">
        <a class="ember-view" data-control-name="background_details_school" href="/search/results/all/?keywords=S.H.S.B.B" id="ember188"> <div class="pv-entity__logo">
          <img alt="S.H.S.B.B" class="pv-entity__logo-img pv-entity__logo-img EntityPhoto-square-4 lazy-image ghost-school ember-view" id="ember190" loading="lazy" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"/>
        </div>
        <div class="pv-entity__summary-info pv-entity__summary-info--background-section">
          <div class="pv-entity__degree-info">
            <h3 class="pv-entity__school-name t-16 t-black t-bold">S.H.S.B.B</h3>
            <!-- --> <p class="pv-entity__secondary-title pv-entity__fos t-14 t-black t-normal">
              <span class="visually-hidden">Field Of Study</span>
              <span class="pv-entity__comma-item">PCM</span>
            </p>
            <!-- --> </div>
            <!-- -->
            <!-- --></div>
          </a>
          <!-- --> </div>
          <!-- --></div>
        </li>
      </ul>
```

We can get the name of the college directly using the `h3` tag.

```python
college_name = edu_section.find('h3').get_text().strip()
college_name
'Technocrats Institute of Technology (Excellence), Anand Nagar, PB No. 24, Post Piplani, BHEL, Bhopal - 462021'
```

We will get the name of the degree from the second `span` of the `p` tag with class `pv-entity__secondary-title pv-entity__degree-name t-14 t-black t-normal`.

```python
degree_name = edu_section.find('p', {'class': 'pv-entity__secondary-title pv-entity__degree-name t-14 t-black t-normal'}).find_all('span')[1].get_text().strip()
degree_name
'Bachelor of Engineering (B.E.)'
```

We will get the stream from the second `span` of the `p` tag with class `pv-entity__secondary-title pv-entity__fos t-14 t-black t-normal`.

```python
stream = edu_section.find('p', {'class': 'pv-entity__secondary-title pv-entity__fos t-14 t-black t-normal'}).find_all('span')[1].get_text().strip()
stream
'Electrical, Electronics and Communications Engineering'
```

We will get the years of degree from the second `span` of the `p` tag with class `pv-entity__dates t-14 t-black--light t-normal`.

```python
degree_year = edu_section.find('p', {'class': 'pv-entity__dates t-14 t-black--light t-normal'}).find_all('span')[1].get_text().strip()
degree_year
'2012 – 2016'
```

We will append everything we have scrapped in `info`.

```python
info
['https://www.linkedin.com/in/rishabh-singh-61b706114/',
 'Rishabh Singh',
 '#futureshaper',
 'Bengaluru, Karnataka, India',
 '500+ connections',
 'Honeywell',
 'FPGA Engineer',
 'Aug 2019 – Present',
 '1 yr 2 mos']
info.append(college_name)
info.append(degree_name)
info.append(stream)
info.append(degree_year)
info
['https://www.linkedin.com/in/rishabh-singh-61b706114/',
 'Rishabh Singh',
 '#futureshaper',
 'Bengaluru, Karnataka, India',
 '500+ connections',
 'Honeywell',
 'FPGA Engineer',
 'Aug 2019 – Present',
 '1 yr 2 mos',
 'Technocrats Institute of Technology (Excellence), Anand Nagar, PB No. 24, Post Piplani, BHEL, Bhopal - 462021',
 'Bachelor of Engineering (B.E.)',
 'Electrical, Electronics and Communications Engineering',
 '2012 – 2016',
 'Technocrats Institute of Technology (Excellence), Anand Nagar, PB No. 24, Post Piplani, BHEL, Bhopal - 462021',
 'Bachelor of Engineering (B.E.)',
 'Electrical, Electronics and Communications Engineering',
 '2012 – 2016']
```

We have scrapped all the important data from the LinkedIn profile. This same code can be used to scrap many more profiles.

**Note:** The IDs and class name of the tags can change. Hence before running this code check the current IDs and class names of the tags used by inspecting the webpage.