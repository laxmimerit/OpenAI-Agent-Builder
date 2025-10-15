https://kgptalkie.com/resume-and-cv-summarization

# Resume and CV Summarization  
**Source:** https://kgptalkie.com/resume-and-cv-summarization  

## Published by  
georgiannacambel  
on  
14 September 2020  

---

## Resume NER Training  

In this blog, we are going to create a model using **SpaCy** which will extract the main points from a resume. We are going to train the model on almost 200 resumes. After the model is ready, we will extract the text from a new resume and pass it to the model to get the summary.  

### Collecting Training Data  
Collecting training data is a very crucial step while building any **machine learning** model. It may sound like an incredibly painful process. In this project, we have used about 200 resumes to train our model.  

You can download the dataset from [here](https://kgptalkie.com/resume-and-cv-summarization).  

### Follow Example Here:  
https://spacy.io/usage/training#training-data  

### Watch Video:  
(Video link not provided in the text)  

---

## Code Implementation  

### Importing Libraries  
```python
import spacy
import pickle
import random
```

### Loading Training Data  
```python
train_data = pickle.load(open('train_data.pkl', 'rb'))
train_data[0]
```

**Example Training Data:**  
```python
('Govardhana K Senior Software Engineer  Bengaluru, Karnataka, Karnataka - Email me on Indeed: indeed.com/r/Govardhana-K/ b2de315d95905b68  Total IT experience 5 Years 6 Months Cloud Lending Solutions INC 4 Month • Salesforce Developer Oracle 5 Years 2 Month • Core Java Developer Languages Core Java, Go Lang Oracle PL-SQL programming, Sales Force Developer with APEX.  Designations & Promotions  Willing to relocate: Anywhere  WORK EXPERIENCE  Senior Software Engineer  Cloud Lending Solutions -  Bangalore, Karnataka -  January 2018 to Present  Present  Senior Consultant  Oracle -  Bangalore, Karnataka -  November 2016 to December 2017  Staff Consultant  Oracle -  Bangalore, Karnataka -  January 2014 to October 2016  Associate Consultant  Oracle -  Bangalore, Karnataka -  November 2012 to December 2013  EDUCATION  B.E in Computer Science Engineering  Adithya Institute of Technology -  Tamil Nadu  September 2008 to June 2012  https://www.indeed.com/r/Govardhana-K/b2de315d95905b68?isid=rex-download&ikw=download-top&co=IN https://www.indeed.com/r/Govardhana-K/b2de315d95905b68?isid=rex-download&ikw=download-top&co=IN   SKILLS  APEX. (Less than 1 year), Data Structures (3 years), FLEXCUBE (5 years), Oracle (5 years), Algorithms (3 years)  LINKS  https://www.linkedin.com/in/govardhana-k-61024944/  ADDITIONAL INFORMATION  Technical Proficiency:  Languages: Core Java, Go Lang, Data Structures & Algorithms, Oracle PL-SQL programming, Sales Force with APEX. Tools: RADTool, Jdeveloper, NetBeans, Eclipse, SQL developer, PL/SQL Developer, WinSCP, Putty Web Technologies: JavaScript, XML, HTML, Webservice  Operating Systems: Linux, Windows Version control system SVN & Git-Hub Databases: Oracle Middleware: Web logic, OC4J Product FLEXCUBE: Oracle FLEXCUBE Versions 10.x, 11.x and 12.x  https://www.linkedin.com/in/govardhana-k-61024944/',
 {'entities': [(1749, 1755, 'Companies worked at'),
   (1696, 1702, 'Companies worked at'),
   (1417, 1423, 'Companies worked at'),
   (1356, 1793, 'Skills'),
   (1209, 1215, 'Companies worked at'),
   (1136, 1248, 'Skills'),
   (928, 932, 'Graduation Year'),
   (858, 889, 'College Name'),
   (821, 856, 'Degree'),
   (787, 791, 'Graduation Year'),
   (744, 750, 'Companies worked at'),
   (722, 742, 'Designation'),
   (658, 664, 'Companies worked at'),
   (640, 656, 'Designation'),
   (574, 580, 'Companies worked at'),
   (555, 573, 'Designation'),
   (470, 493, 'Companies worked at'),
   (444, 469, 'Designation'),
   (308, 314, 'Companies worked at'),
   (234, 240, 'Companies worked at'),
   (175, 198, 'Companies worked at'),
   (93, 137, 'Email Address'),
   (39, 48, 'Location'),
   (13, 38, 'Designation'),
   (0, 12, 'Name')]})
```

---

## Training the Model  

### Code for Model Training  
```python
nlp = spacy.blank('en')

def train_model(train_data):
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    for _, annotation in train_data:
        for ent in annotation['entities']:
            ner.add_label(ent[2])
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(10):
            print("Statring iteration " + str(itn))
            random.shuffle(train_data)
            losses = {}
            index = 0
            for text, annotations in train_data:
                try:
                    nlp.update(
                        [text],
                        [annotations],
                        drop=0.2,
                        sgd=optimizer,
                        losses=losses)
                except Exception as e:
                    pass
            print(losses)
train_model(train_data)
```

### Training Output  
```
Statring iteration 0
{'ner': 15037.69734973145}
Statring iteration 1
{'ner': 14219.407121717988}
Statring iteration 2
{'ner': 10966.300542596338}
Statring iteration 3
{'ner': 9323.783655686213}
Statring iteration 4
{'ner': 8716.684299001408}
Statring iteration 5
{'ner': 7539.511302326008}
Statring iteration 6
{'ner': 5957.294807362359}
Statring iteration 7
{'ner': 4725.2797832249025}
Statring iteration 8
{'ner': 5975.5969295029045}
Statring iteration 9
{'ner': 4244.961263079282}
```

### Saving the Model  
```python
nlp.to_disk('nlp_model')
```

---

## Testing the Model  

### Loading the Model  
```python
nlp_model = spacy.load('nlp_model')
```

### Example Test on Training Data  
```python
train_data[0][0]
'Vineeth Vijayan "Store Executive" - Orange City Hospital & Research Institute  Nagpur, Maharashtra, Maharashtra - Email me on Indeed: indeed.com/r/Vineeth-Vijayan/ ee84e7ea0695181f  I have over 2 years of experience working as a "Store Executive" in a reputed big organisation based in India. I am a hard worker and a very good learner. I aspire to grow in my career by joining a good organisation.  I am good at work and people management. I possess good computer knowledge. I have underwent training in courses like MS Excel Level 1, 2 &amp; 3. MS Word &amp; Powerpoint. I have experience working in ERP. I am a certified ISO 27001: 2013 Internal Auditor.  Willing to relocate: Anywhere  WORK EXPERIENCE  "Store Executive"  Orange City Hospital & Research Institute -  Nagpur, Maharashtra -  March 2016 to Present  Job Profile: - 1. Keep records to maintain inventory control, cost containment and to assure proper stock levels. 2. Rotate stock and arrange for disposal of surpluses. 3. Plan and perform work that involves ordering, receiving, inspecting, returning, unloading, shelving, packing, labeling and maintaining a perpetual inventory of items. 4. Coordinate freight handling, equipment moving and minor repairs. 5. Perform peripheral site-specific duties as required and instructed by supervisors. 6. Monitoring and updating stock details etc. in Mednet store software. Use ABC, VED &amp; FSN Stock Analysis methods when required. 7. Report any stock discrepancy to supervisor for appropriate action.  "HR Officer "  Gulf Warehousing Company Qatar - Agility Group -  January 2012 to July 2015  a multinational company catering Customers in the field of Warehousing, Freight Forwarding, Records Management, Courier and International Relocation Services.  Job Profile: - 1. Deliver new hires orientation program to enable new hires to adjust themselves to new working environment and facilitating newcomers joining formalities. 2. Design and develop Training Calendar and budget for all type of training programs, for the year ahead for all designations within the company, according to the needs of the business and Training Need Analysis survey results.  https://www.indeed.com/r/Vineeth-Vijayan/ee84e7ea0695181f?isid=rex-download&ikw=download-top&co=IN https://www.indeed.com/r/Vineeth-Vijayan/ee84e7ea0695181f?isid=rex-download&ikw=download-top&co=IN   3. Sourcing and coordinating with external vendors for training that are requested by line managers/departments. 4. Maintaining employee training records and updating it on a regular basis. 5. Assist Supervisors in developing and implementing changes in HR policies and procedures. 6. Coordinate in conducting employee engagement programs. 7. Support departments in complying with ISO Quality Management requirements and facilitate various HR related welfare activities.  "Training Coordinator"  M-Squared Software & Services - Kerala -  Kerala, IN -  July 2010 to December 2011  a US based company with headquarters in Nevada, Las Vegas that deals with security gadgets, medical transcription and software development.  Job Profile: - 1. Perform regular office administration duties that includes: Office infrastructure management, housekeeping supervision, responding and sending e-mails to head office and clients, monitoring time-in &amp; time-out of employees, managing employee leave status and office cash management. 2. Handle walk-in enquires, attend telephone calls and conduct course induction sessions for new joiners. 3. Train the students in work related processes. 4. Anchoring and conducting course related seminars in well-established educational institutions. 5. Manage training materials that are stipulated within the company regulations. 6. Formulating training programs and sending it to training manager and to all concerned departments for approval. 7. Preparing tools for training. 8. Organize the training that includes: distribution of training materials, monitor trainees in training and support trainers in training. 9. Record training plans and training programs. 10. Maintain employees and students database. 11. Record results of evaluation and training. 12. Archive records of training, including course content, the training, number of students, results, and feedback. 13. Conduct team huddles/counseling sessions for trainees when required.  "Manager"  Nair IT Pvt. Ltd -  Nagpur, Maharashtra -  September 2008 to July 2010  Client: Accentia Oak Technologies.  Job Profile: - 1. To manage the day-to-day planning, operation and problem-solving of the team to meet the required service level components, quality and productivity targets of the client.    2. Delivery of team statistics, quality and productivity targets &amp; indicators, taking note any downtime during the day and report it to the management level superiors. 3. Operational Management: Managing the floor, adherence to schedule and monitoring of attendance. Submits daily report to the management level superiors. 4. Monitoring the portal periodically, prioritizing jobs according to TAT, and assigning jobs to correct work pool. 5. Coaching and relaying feedback to MTs/Editors. 6. To motivate the team to achieve the client targets. Update the team on the status of work at the start and end of each shift. 7. Compiling reports on team\'s performance and client feedback.  "Process Executive" in Infosys BPO Limited, Bangalore  CISCO -  Bengaluru, Karnataka -  January 2006 to July 2008  Job Profile: - 1. To handle end to end order placement process of Cisco. 2. Book orders for customers as per Cisco\'s booking policies. 3. Customer advocacy. 4. Check and avail special discounts, services, licensing, upgrades etc., to the customers. 5. Receive customer feedback on products, services etc., and resolve them if any. 6. To work along with logistic team to ensure that the products reach precise destinations safely. 7. Provide round the clock 24/7 customer service to the client.  "Sr. Medical Transcriptionist"  DTS Information Systems Pvt. Ltd -  Bengaluru, Karnataka -  December 2004 to January 2006  in DTS Information Systems Pvt. Ltd, Bangalore (A wholly owned subsidiary of DTS America Inc., Tennessee, USA) from Dec 2004 - Jan 2006. Client: Vanderbilt University Medical Center, Nashville, Tennessee.  Job Profile: - 1. To transcribe documents pertaining to patient\'s summary of illness, diseases and diagnostic procedures etc. to avail medical insurance claims as per American transcription guidelines. 2. To ensure that finished end document meets the clients TAT and Quality requirements. 3. To counsel MT\'s regarding their errors etc and if needed conduct team huddles. Computer Exposure  Operating System: Windows XP Microsoft Office Package: MS Excel, MS Word, MS Access and Power Point. Experience in working on Oracle.  ❖ Completed MS Excel Level 1, 2 &amp; 3 Courses from New Horizons Computer Training Center, Qatar.    EDUCATION  B.Sc. in Microbiology in Microbiology  S.N.G College, Bharathiar University  April 2004  SKILLS  Storekeeping / Inventory Management / Purchase (2 years), People & Time Management (2 years)  ADDITIONAL INFORMATION  Competencies  ❖ Effective communications skills, excellent interpersonal skills, good time management skills, results-oriented individual and a very good team player. ❖ Sincere, hardworking and willing to take up challenging assignments.  Other Qualifications  ❖ ISO 27001: 2013 Internal Auditor Certification from Coms Vantage Consultancy, Qatar in October 2014 ❖ ISO 27001: 2013 Awareness Certification from Coms Vantage Consultancy, Qatar in September 2014 ❖ Carriage of Dangerous Goods by Road Certification in 2012 from IRU Academy in tie up with Mowasalat - State of Qatar owned Transport Company. ❖ First class in Hindi Parichaya Examination conducted by Dakshina Bharat Hindi Prachar Sabha. ❖ Second class in Prathamic Examination conducted by Dakshina Bharat Hindi Prachar Sabha.'
```

### Model Output  
```python
doc = nlp_model(train_data[0][0])
for ent in doc.ents:
    print(f'{ent.label_.upper():{30}}- {ent.text}')
```

**Output:**  
```
NAME                          - Vineeth Vijayan
DESIGNATION                   - Store Executive
EMAIL ADDRESS                 - indeed.com/r/Vineeth-Vijayan/ ee84e7ea0695181f
DESIGNATION                   - Store Executive
DEGREE                        - B.Sc. in Microbiology in Microbiology
COLLEGE NAME                  - S.N.G College
SKILLS                        - Storekeeping / Inventory Management / Purchase (2 years), People & Time Management (2 years)
```

---

## Testing on Unseen Resume  

### Extracting Text from PDF  
```python
import sys, fitz
fname = 'Alice Clark CV.pdf'
doc = fitz.open(fname)
text = ""
for page in doc:
    text = text + str(page.getText())
tx = " ".join(text.split('\n'))
print(tx)
```

**Extracted Text:**  
```
Alice Clark  AI / Machine Learning    Delhi, India Email me on Indeed  •  20+ years of experience in data handling, design, and development  •  Data Warehouse: Data analysis, star/snow flake scema data modelling and design specific to  data warehousing and business intelligence  •  Database: Experience in database designing, scalability, back-up and recovery, writing and  optimizing SQL code and Stored Procedures, creating functions, views, triggers and indexes.  Cloud platform: Worked on Microsoft Azure cloud services like Document DB, SQL Azure,  Stream Analytics, Event hub, Power BI, Web Job, Web App, Power BI, Azure data lake  analytics(U-SQL)  Willing to relocate anywhere    WORK EXPERIENCE  Software Engineer  Microsoft – Bangalore, Karnataka  January 2000 to Present  1. Microsoft Rewards Live dashboards:  Description: - Microsoft rewards is loyalty program that rewards Users for browsing and shopping  online. Microsoft Rewards members can earn points when searching with Bing, browsing with  Microsoft Edge and making purchases at the Xbox Store, the Windows Store and the Microsoft  Store. Plus, user can pick up bonus points for taking daily quizzes and tours on the Microsoft  rewards website. Rewards live dashboards gives a live picture of usage world-wide and by  markets like US, Canada, Australia, new user registration count, top/bottom performing rewards  offers, orders stats and weekly trends of user activities, orders and new user registrations. the  PBI tiles gets refreshed in different frequencies starting from 5 seconds to 30 minutes.  Technology/Tools used    EDUCATION  Indian Institute of Technology – Mumbai  2001    SKILLS  Machine Learning, Natural Language Processing, and Big Data Handling    ADDITIONAL INFORMATION  Professional Skills  • Excellent analytical, problem solving, communication, knowledge transfer and interpersonal  skills with ability to interact with individuals at all the levels  • Quick learner and maintains cordial relationship with project manager and team members and  good performer both in team and independent job environments  • Positive attitude towards superiors &amp; peers  • Supervised junior developers throughout project lifecycle and provided technical assistance
```

### Model Output  
```python
doc = nlp_model(tx)
for ent in doc.ents:
    print(f'{ent.label_.upper():{30}}- {ent.text}')
```

**Output:**  
```
NAME                          - Alice Clark
LOCATION                      - Delhi
DESIGNATION                   - Stream Analytics
DESIGNATION                   - Software Engineer
COMPANIES WORKED AT           - Microsoft –
DEGREE                        - Indian Institute of Technology – Mumbai
SKILLS                        - Machine Learning, Natural Language Processing, and Big Data Handling    ADDITIONAL INFORMATION  Professional Skills  • Excellent analytical, problem solving, communication, knowledge transfer and interpersonal
```

---

## Conclusion  
To get a better and accurate summary, you can train the model on more data samples. You can include different kinds of resumes in the training samples.  

**Source:** https://kgptalkie.com/resume-and-cv-summarization  
**Source:** https://kgptalkie.com/resume-and-cv-summarization