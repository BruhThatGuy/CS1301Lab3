import streamlit as st
import info3
import pandas as pd
#about me
def about_me_section():
    st.header("About Me")
    st.image("https://github.com/lhernstberger/CS1301Lab3/blob/ab17a4fece27f64d37ce50f9a65d65c2b7162b2c/Lab3/Images/unknown.jpeg", width=200)
        
    #info3.profile_picture, width = 200)
    st.write(info3.about_me)
    st.write("Go Jackets!")
    st.write("---")
about_me_section()


#sidebar links
def links_section():
    st.sidebar.header("Links")
    st.sidebar.text("See my cinema taste on Letterboxd")
    letterboxd_link=f'<a href="{info3.my_letterboxd_url}"><img src="{info3.letterboxd_image_url}" alt="letterboxd" width ="75" height ="75"></a>'
    st.sidebar.markdown(letterboxd_link, unsafe_allow_html=True)
    st.sidebar.text("Checkout my work")
    github_link = f'<a href="{info3.my_github_url}"><img src="{info3.github_image_url}" alt ="Github" width="65" height="65"></a>'
    st.sidebar.markdown(github_link, unsafe_allow_html=True)
    st.sidebar.text("Or email me!")
    email_html = f'<a href = "mailto:{info3.my_email_address}"><img src="{info3.email_image_url}" alt ="Email" width ="75" height ="75"></a>'
    st.sidebar.markdown(email_html, unsafe_allow_html=True)
links_section()


# education
def education_section(education_data, education_data_two, course_data):
    st.header("Education")
    st.subheader(f"**{education_data['Institution']}**")
    st.write(f"**Degree:** {education_data['Degree']}")
    st.write(f"**Graduation Date:** {education_data['Graduation Date']}")
    st.write(f"**GPA::** {education_data['GPA']}")
    st.subheader(f"**{education_data_two['Institution']}**")
    st.write(f"**Degree:** {education_data_two['Degree']}")
    st.write(f"**Graduation Date:** {education_data_two['Graduation Date']}")
    st.write(f"**GPA::** {education_data_two['GPA']}")
    st.subheader("**Relevant Coursework:**")
    coursework = pd.DataFrame(course_data)
    st.dataframe(coursework, column_config={
        "chem":"Course Code",
        "names": "Course Names",
        "semester_taken": "Semester Taken",
        "skills": "What I Learned"},
        hide_index=True,
        )
    
    st.write("---")
education_section(info3.education_data, info3.education_data_two, info3.course_data)

#Professional Experience

def experience_section(experience_data):
    st.header("Professional Experience")
    for job_title, (job_description, image) in experience_data.items():
        expander=st.expander(f"{job_title}")
        expander.image(image, width=250)
        for bullet in job_description:
            expander.write(bullet)
    st.write("---")
experience_section(info3.experience_data)

#Projects

def project_section(projects_data):
    st.header("Projects")
    for project_name, project_description in projects_data.items():
        expander=st.expander(f"{project_name}")
        expander.write(project_description)
    st.write("---")
project_section(info3.projects_data)

#skills

def skills_section(programming_data, spoken_data):
    st.header("Skills")
    st.subheader("Programming Languages")
    for skill, percentage in programming_data.items():
        st.write(f"{skill} {info3.programming_icons.get(skill, )}")
        st.progress(percentage)
    st.subheader("Spoken Languages")
    for spoken, proficiency in spoken_data.items():
        st.write(f"{spoken} {info3.spoken_icons.get(spoken, )}: {proficiency}")
    st.write("---")
skills_section(info3.programming_data, info3.spoken_data)

#activities

def activities_section(leadership_data, activity_data):
    st.header("Activities")
    tab1,tab2 = st.tabs(["Leadership","Community"])
    with tab1:
        st.subheader("Leadership")
        for title, (details, image) in leadership_data.items():
            expander=st.expander(f"{title}")
            expander.image(image, width=250)
            for bullet in details:
                expander.write(bullet)
    with tab2:
        st.subheader("Community Service")
        for title, details in activity_data.items():
            expander=st.expander(f"{title}")
            for bullet in details:
                expander.write(bullet)
    st.write("---")
activities_section(info3.leadership_data, info3.activity_data)
        
