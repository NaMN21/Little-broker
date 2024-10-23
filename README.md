# LittleBroker


## Project Overview

This project started with an attempt to perform web scraping on **Dubizzle Egypt** for real estate data. However, due to the way the company's website was built, it was challenging to extract clear and usable data.

### Data Collection
After encountering issues with Dubizzle, **Bilal** collaborated with **Nassar** to collect real estate data from the **Aqarmap** website. This alternative source provided the necessary data more effectively, allowing us to proceed with the project.

### Frontend Development
Once the data was gathered, **Bilal** and **Ali** started working on developing the user interface using **Streamlit**. This allowed for a simple and interactive frontend for users to engage with the application.

### Model Integration
The frontend was connected to an API-powered **LLama model** using the Hugging Face model hub. The application generates real estate-related text using Llama,providing users with dynamically generated content.

### GitHub and CI/CD Setup
The project was uploaded to **GitHub**, and **GitHub Actions** was implemented to ensure continuous integration and deployment. This setup allows the application to automatically update whenever changes are pushed to the repository, ensuring the live version is always in sync with the latest code.

### Streamlit Deployment
The final version of the application was deployed on **Streamlit**, making it publicly accessible for users to interact with.

[Visit the Application]