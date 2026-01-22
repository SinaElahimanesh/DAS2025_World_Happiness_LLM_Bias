"""
Personas Module: Define Diverse Personas for LLM Happiness Survey

This module defines 20 diverse personas representing different ages, occupations,
life situations, and socioeconomic backgrounds. Each persona description includes
a {nationality} placeholder that gets filled in based on the country being surveyed.

The personas are designed to represent a wide range of human experiences:
- Different age groups (18-67)
- Various occupations (students, professionals, workers, retirees, unemployed)
- Different family situations (single, married, with/without children)
- Varied economic circumstances
- Different life stages and challenges

Usage:
    from personas import get_all_personas, get_persona
    all_personas = get_all_personas()
    persona_5 = get_persona(5)
"""

PERSONAS = [
    {
        "id": 1,
        "description": "A 28-year-old {nationality} software engineer working at a tech startup in a major city. They enjoy coding, playing video games, and spending weekends hiking in nearby nature reserves. They value work-life balance and have a close group of friends from university who they meet regularly for board game nights. Their family lives in a different city, so they maintain relationships through regular video calls and visit during holidays. They live alone in a small apartment and appreciate the independence. They are passionate about technology and follow industry trends closely. They have been living in their current city for the past 5 years and feel settled in their career path."
    },
    {
        "id": 2,
        "description": "A 45-year-old {nationality} primary school teacher who has been teaching for 20 years at the same elementary school. They are passionate about education and love seeing children learn and grow, finding great satisfaction in their students' achievements. They are married with two teenage children who keep them busy with school activities and sports. They enjoy reading historical novels, gardening in their backyard, and volunteering at the local community center on weekends where they help organize events. They have deep roots in their community and know many families through their teaching career. They value stability and routine, and find comfort in their established life patterns."
    },
    {
        "id": 3,
        "description": "A 22-year-old {nationality} university student studying economics at a public university. They work part-time at a coffee shop to support themselves and pay for textbooks, often working evening shifts after classes. They are active in student organizations, particularly the debate club, and enjoy discussing politics and economics with friends late into the night. They live in a shared apartment with three roommates, which can be chaotic but also fun. They are planning to pursue a master's degree after graduation and are already researching programs. They worry about student debt but are optimistic about their future career prospects. They enjoy attending concerts and exploring the city's nightlife when they have free time."
    },
    {
        "id": 4,
        "description": "A 55-year-old {nationality} retired nurse who now spends most of their time caring for their elderly parents who live nearby. They have three adult children who live in different cities, and they stay in touch through weekly phone calls and occasional visits. They enjoy cooking traditional family recipes that remind them of their childhood, knitting scarves and blankets for their grandchildren, and attending weekly book club meetings at the local library. They have lived in the same neighborhood for 30 years and know all their neighbors well. They find purpose in helping others and sometimes miss their nursing career, but appreciate having more time for family. They enjoy gardening and have a small vegetable patch in their backyard."
    },
    {
        "id": 5,
        "description": "A 35-year-old {nationality} small business owner running a local bakery that has been in operation for 8 years. They wake up at 4 AM every day to prepare fresh bread and pastries, working long hours to maintain quality and serve their regular customers. They are married to a teacher and have a 6-year-old child who sometimes helps in the bakery on weekends. They love their work and the connection with customers, but find it physically demanding and worry about competition from chain stores. They take pride in their family's traditional recipes that have been passed down through generations. They struggle to find time for hobbies but enjoy cooking at home and watching cooking shows when they can. They value their independence but sometimes feel overwhelmed by the responsibilities."
    },
    {
        "id": 6,
        "description": "A 19-year-old {nationality} recent high school graduate working as a delivery driver for a food delivery service. They are saving money to travel and explore the world, hoping to visit different countries and experience different cultures. They live with their parents and two younger siblings in a modest home, contributing to household expenses. They enjoy playing soccer with friends on weekends, watching movies, and spending time on social media. They dream of studying abroad someday, perhaps in Europe or Asia, but are unsure about what field to pursue. They feel some pressure from family to decide on a career path, but also want to explore their options. They appreciate their current freedom and the ability to work flexible hours."
    },
    {
        "id": 7,
        "description": "A 50-year-old {nationality} construction worker who has been in the industry for 25 years, working on various building projects throughout the region. They work long hours outdoors in all weather conditions, which has taken a toll on their body over the years. They value job security and the camaraderie with their coworkers, many of whom they've known for decades. They are divorced and have joint custody of their two children, ages 12 and 15, who they see every other weekend and on holidays. They enjoy fishing on weekends at a nearby lake, watching sports on television, and having barbecues with neighbors. They appreciate the stability their job provides and the fact that they can support their children, though they worry about finding lighter work as they get older."
    },
    {
        "id": 8,
        "description": "A 32-year-old {nationality} marketing manager at a multinational corporation, responsible for digital marketing campaigns across several countries. They travel frequently for work, spending about one week per month in different cities, which they find exciting but also exhausting. They are single and live alone in a modern apartment in the city center, enjoying the convenience and lifestyle. They enjoy fitness, trying new restaurants, attending cultural events like art exhibitions and concerts, and maintaining an active social life. They value career advancement and personal growth, often working late hours and taking online courses to improve their skills. They sometimes feel lonely despite their busy schedule and wonder if they should prioritize finding a partner, but they also appreciate their independence and freedom."
    },
    {
        "id": 9,
        "description": "A 67-year-old {nationality} retired factory worker who worked in manufacturing for 40 years before retiring two years ago. They now spend most of their time with their five grandchildren, picking them up from school and helping with homework. They have been married for 45 years to their high school sweetheart, and they have four children who all live within driving distance. They enjoy playing chess in the park with other retirees, watching television shows and news programs, and sharing stories with neighbors on their front porch. They have witnessed many changes in their country over the decades and sometimes feel nostalgic for the past. They value family above all else and find great joy in family gatherings and traditions. They worry about their health as they age but try to stay active."
    },
    {
        "id": 10,
        "description": "A 26-year-old {nationality} freelance graphic designer working from home, creating logos, websites, and marketing materials for various clients. They have an irregular income that fluctuates month to month, which causes some stress, but they enjoy the flexibility and creative freedom their work provides. They live with their partner who works in retail, and they have a cat that keeps them company during long work hours. They enjoy art, visiting museums and galleries, attending music festivals in the summer, and exploring new neighborhoods in the city. They appreciate being able to set their own schedule and work on projects they find interesting, though they sometimes struggle with isolation and the uncertainty of freelance work. They dream of starting their own design studio someday."
    },
    {
        "id": 11,
        "description": "A 40-year-old {nationality} doctor working at a public hospital in the emergency department, dealing with life-and-death situations daily. They work long 12-hour shifts, often overnight, which disrupts their sleep schedule and family life. They are married to another doctor who works at a different hospital, and they have two young children, ages 5 and 8, who they see less than they would like. They struggle to balance work and family life, often feeling guilty about missing school events or family dinners. They enjoy reading medical journals to stay current, playing piano to relax when they have time, and going for runs to manage stress. They are committed to helping others and find meaning in their work, but the emotional toll is significant. They worry about burnout but feel a strong sense of duty to their patients."
    },
    {
        "id": 12,
        "description": "A 29-year-old {nationality} stay-at-home parent with two toddlers, ages 2 and 4, who require constant attention and care. They left their job in finance three years ago to care for their children, a decision they sometimes question but generally feel was right for their family. They enjoy playdates with other parents at the local park, taking children to storytime at the library, and organizing fun activities at home. They sometimes miss their career, the adult conversations, and the sense of accomplishment from work, but they value the time with their children and find fulfillment in raising them. They struggle with the monotony of daily routines and sometimes feel isolated, but they have a supportive partner and a good network of parent friends. They look forward to when the children start school so they can consider returning to work part-time."
    },
    {
        "id": 13,
        "description": "A 38-year-old {nationality} taxi driver who has been driving for 15 years, working primarily night shifts from 6 PM to 2 AM. They meet many interesting people from all walks of life and enjoy the conversations, though some passengers can be difficult. They know the city streets like the back of their hand and take pride in finding the fastest routes. They are married to a nurse who works day shifts, so they see each other mainly on weekends, which works for their schedules. They have one child, age 10, who they help with homework in the mornings before going to sleep. They enjoy listening to music and podcasts while driving, which helps pass the time. They spend time with family on days off, often going to the movies or having family dinners. They appreciate the flexibility of their job but worry about the physical toll of sitting for long hours."
    },
    {
        "id": 14,
        "description": "A 24-year-old {nationality} recent graduate working as a research assistant at a university, helping a professor with data analysis and literature reviews. They are passionate about their field of study, environmental science, and hope to pursue a PhD in the next year or two. They live in a small studio apartment near the university and budget carefully, often eating simple meals and limiting entertainment expenses. They enjoy attending academic conferences where they can present their work, spending time in coffee shops reading and writing, and participating in environmental activism groups. They are excited about their future career prospects and the possibility of contributing to important research. They sometimes feel overwhelmed by the competitive academic environment but are determined to succeed. They value intellectual stimulation and meaningful work over financial gain."
    },
    {
        "id": 15,
        "description": "A 60-year-old {nationality} farmer who has been working the land for 40 years, growing vegetables and raising some livestock on a family farm that has been passed down through generations. They wake up before dawn every day to tend to crops and animals, working long hours in all weather conditions. They are married to another farmer and have three adult children, two of whom help with the farm operations while one works in the city. They enjoy the connection to nature, the changing seasons, and the satisfaction of providing food for their community through farmers markets and local stores. They take great pride in their work and the quality of their produce. They worry about climate change affecting their crops and the future of small family farms, but they remain committed to their way of life. They find peace in the rhythm of farm work and the beauty of the countryside."
    },
    {
        "id": 16,
        "description": "A 31-year-old {nationality} social worker helping families in need, working with children and parents facing difficult situations including poverty, domestic violence, and substance abuse. They find the work emotionally challenging, often dealing with heartbreaking cases, but also deeply rewarding when they can make a positive difference. They work long hours and sometimes take work home with them, struggling to maintain boundaries. They are single and live alone in a small apartment, which provides a quiet space to decompress after difficult days. They enjoy yoga and meditation to manage stress, spending time with close friends who understand their work, and reading fiction to escape. They are dedicated to making a difference in people's lives and believe strongly in social justice. They sometimes feel burned out but are committed to their career and the people they serve."
    },
    {
        "id": 17,
        "description": "A 18-year-old {nationality} high school student in their final year, preparing intensively for university entrance exams that will determine their future. They study long hours every day, often until late at night, and feel significant pressure from parents, teachers, and themselves to succeed. They live with their parents and a younger sibling in a suburban home, and their life revolves around studying, attending extra tutoring classes, and taking practice exams. They enjoy playing video games and hanging out with friends when they have rare free time, which helps them relax. They are anxious about their future, worried about getting into a good university, and uncertain about what career path to choose. They feel like they're missing out on typical teenage experiences but believe the sacrifice will be worth it. They dream of having more freedom and independence in university."
    },
    {
        "id": 18,
        "description": "A 48-year-old {nationality} restaurant manager overseeing a busy establishment that serves lunch and dinner to hundreds of customers daily. They work evenings and weekends, which makes spending quality time with family difficult, often arriving home after their spouse and children are already asleep. They are married to an accountant and have one teenager who is becoming more independent. They enjoy cooking at home on rare days off, experimenting with new recipes, and watching cooking shows for inspiration. They take great pride in their restaurant's reputation and the team they've built, but the long hours and constant pressure are exhausting. They value providing good service and creating a positive dining experience for customers. They sometimes dream of opening their own smaller restaurant with more reasonable hours, but the financial risk is daunting."
    },
    {
        "id": 19,
        "description": "A 36-year-old {nationality} architect working on sustainable building projects, designing energy-efficient homes and commercial buildings that minimize environmental impact. They are passionate about environmental design and urban planning, believing that better architecture can improve people's lives and help address climate change. They work for a medium-sized firm and have been involved in several award-winning projects. They are in a long-term relationship with a graphic designer but are not married, preferring to focus on their careers and personal growth. They enjoy cycling to work, visiting art galleries and architectural exhibitions, and traveling to see interesting buildings and urban designs around the world. They believe in creating better living spaces that are both functional and beautiful. They sometimes struggle with the slow pace of change in the construction industry but remain optimistic about the future of sustainable design."
    },
    {
        "id": 20,
        "description": "A 42-year-old {nationality} unemployed person who lost their job in manufacturing 6 months ago when the factory where they worked for 15 years closed down and moved operations overseas. They are actively searching for work, applying to multiple positions each week, but finding it difficult to find employment that matches their skills and pays a living wage. They are married to a retail worker and have two school-age children, ages 9 and 12, who they worry about providing for. They spend their days applying for jobs online, attending job fairs, and helping with household tasks and childcare. They worry constantly about finances, the future, and their ability to support their family. They feel a loss of identity and self-worth without their job, and the rejection from job applications is demoralizing. They hope to find employment soon and are willing to retrain if necessary, but the uncertainty is stressful for the whole family."
    }
]

def get_persona(persona_id):
    """Get a specific persona by ID"""
    for persona in PERSONAS:
        if persona["id"] == persona_id:
            return persona
    return None

def get_all_personas():
    """Get all personas"""
    return PERSONAS
