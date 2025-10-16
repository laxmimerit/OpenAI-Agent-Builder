# Zapier MySQL Lambda API Setup Guide

Complete step-by-step guide to create a GET API for MySQL database access via AWS Lambda and API Gateway.

---

## Prerequisites

- AWS Account with Lambda and API Gateway access
- RDS MySQL database (publicly accessible or in VPC)
- Database credentials (host, username, password, database name)
- Terminal/Command Prompt access

---

## Part 1: Locate PyMySQL Layer ZIP
pymysql-layer.zip 

## Part 2: Upload PyMySQL Layer to AWS

1. Go to AWS Lambda Console
2. Click **Layers** in left sidebar
3. Click **Create layer**
4. Configuration:
   - Name: `pymysql-layer`
   - Description: PyMySQL library for Lambda
   - Upload: Select your `pymysql-layer.zip` file
   - Compatible runtimes: Select `Python 3.13` (or your version)
5. Click **Create**

---

## Part 3: Create Lambda Function

1. Go to AWS Lambda Console
2. Click **Create function**
3. Configuration:
   - Function name: `rds`
   - Runtime: `Python 3.13`
   - Architecture: `x86_64`
   - Permissions: Create new role with basic Lambda permissions
4. Click **Create function**

---

## Part 4: Add PyMySQL Layer to Lambda

1. In your Lambda function, scroll to **Layers** section
2. Click **Add a layer**
3. Choose one:
   - **Custom layers**: Select your `pymysql-layer` (if you uploaded)
   - **Specify an ARN**: Paste public layer ARN
4. Click **Add**

---

## Part 5: Add Lambda Code

1. In Lambda function, go to **Code** tab
2. Replace all code in `lambda_function.py` with the GET API code
3. Click **Deploy**
4. Wait for "Successfully deployed" message

---

## Part 6: Configure Environment Variables

1. Click **Configuration** tab
2. Click **Environment variables** (left menu)
3. Click **Edit**
4. Add these variables:
   - Key: `DB_HOST` | Value: Your RDS endpoint
   - Key: `DB_USER` | Value: Your MySQL username
   - Key: `DB_PASSWORD` | Value: Your MySQL password
   - Key: `DB_NAME` | Value: Your database name
   - Key: `DB_PORT` | Value: `3306`
5. Click **Save**

---

## Part 7: Increase Lambda Timeout

1. Stay in **Configuration** tab
2. Click **General configuration** (left menu)
3. Click **Edit**
4. Change **Timeout** from `3 seconds` to `30 seconds`
5. Click **Save**

---

## Part 8: Test Lambda Function

1. Go to **Test** tab
2. Click **Create new event**
3. Event name: `test-select`
4. Replace JSON with:
   ```json
   {
     "queryStringParameters": {
       "query": "SELECT * FROM customers LIMIT 5"
     }
   }
   ```
5. Click **Save**
6. Click **Test**
7. Check for success response with data

---

## Part 9: Create API Gateway (HTTP API)

1. Go to **API Gateway** Console
2. Click **Create API**
3. Choose **HTTP API**
4. Click **Build**

### Add Integration:
- Click **Add integration**
- Integration type: **Lambda**
- AWS Region: Select your region
- Lambda function: Select `rds`
- Version: `2.0`

### API Settings:
- API name: `zapier-mysql-api`
- Click **Next**

---

## Part 10: Configure Routes

1. Method: Select **GET**
2. Resource path: `/query`
3. Integration target: Your Lambda (auto-selected)
4. Click **Next**

---

## Part 11: Configure Stage

1. Stage name: `$default` (or `prod`)
2. Check **Auto-deploy**
3. Click **Next**
4. Review settings
5. Click **Create**

---

## Part 12: Get API Invoke URL

1. After creation, go to **Stages** (left sidebar)
2. Click on `$default`
3. Copy the **Invoke URL**
   - Example: `https://abc123xyz.execute-api.us-east-1.amazonaws.com`

Your full endpoint:
```
https://abc123xyz.execute-api.us-east-1.amazonaws.com/query
```

---

## Part 13: Test API

### Test in Browser:
Open this URL (replace with your actual API URL):
```
https://YOUR-API-ID.execute-api.REGION.amazonaws.com/query?query=SELECT * FROM customers LIMIT 3
```

### Test with curl (Linux/Mac/Git Bash):
```bash
curl -k "https://YOUR-API-ID.execute-api.REGION.amazonaws.com/query?query=SELECT * FROM customers LIMIT 3"
```

### Test with PowerShell (Windows):
```powershell
Invoke-RestMethod -Uri "https://YOUR-API-ID.execute-api.REGION.amazonaws.com/query?query=SELECT * FROM customers LIMIT 3"
```
