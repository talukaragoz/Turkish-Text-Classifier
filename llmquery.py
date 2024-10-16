from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage

llm = ChatOllama(
    model="llama3",
    temperature=0
)

def test_llm(text, label):
    messages = [
        (
            "system",
            "You are a classifier of Turkish text excerpts into certain domains.\
                You will be provided with an excerpt of text and will have to\
                place the text into one of these domains:\n\
                astronomy, government, law, location, tv, computer, business, \
                film, medicine, military, time, organization, people, sports, \
                soccer, architecture, geography, music, games, royalty, education, \
                award, biology, internet, symbols, book, media_common, visual_art, \
                travel, fictional_universe, aviation, transportation, chemistry, \
                language, finance, automotive, opera, comic_books, basketball, \
                food, interests, theater, religion, measurement_unit, fashion, \
                meteorology, engineering, exhibitions, physics \n\
                ---------------\n\
                You will be given a text excerpt by the user in Turkish and will \
                have to output just the name of the domain you think applies most \
                to the excerpt. Below are examples of message-output pairs that are \
                correct:\n\
                ---------------\n\
                'message': Laborie , Saint Lucia'yı oluşturan 11 idarî şehrinden biridir\
                'output': location\n\
                ---------------\n\
                'message': İnci Küpeli Kız romanı ve bu romandan uyarlanan aynı isimli \
                film Vermeer'in İnci Küpeli Kız tablosundaki modeliyle olan ilişkisini \
                anlatır\
                'output': visual_art\n\
                ---------------\n\
                These examples were correct as they only output the domain most closely \
                associated with the excerpt. Here are two examples NOT to follow:\n\
                'message': Laborie , Saint Lucia'yı oluşturan 11 idarî şehrinden biridir\
                'output': place\n\
                - place is not a domain, even though it is synonymous to location. Do not\
                make up new domain names, use the ones listed\n\
                'message': Süre , Snow Leopard\'ın bağımsız perakende sürümü için lisans \
                Mac OS X v10.5 \" Leopard \" dan kullanıcılar için yükseltmeyi kısıtlıyor\
                'output': computer because Mac OSX Leopard is associated with Mac computers\n\
                - While the domain is correct, we do not want a reasoning, rather just the \
                name of the domain.\n\
                ---------------\n\
                Use the examples given to output existing domain names you most closely \
                associate with the excerpts. PLEASE ANSWER WITH A SINGLE WORD! Do NOT add \
                any artifacts such as a new line, DO NOT ADD SPACES!, \
                just the domain name from the list provided. Do NOT even provide the \
                'output' tag I added, that is for presenting purposes. Do not start \
                your answer with output:"
        ),
        (
            "human", 
            text),
    ]
    out = llm.invoke(messages).content
    return out, label in out
    

x = input("Query from the user:\n")

messages = [
    (
        "system",
        "You are a classifier of Turkish text excerpts into certain domains.\
            You will be provided with an excerpt of text and will have to\
            place the text into one of these domains:\n\
            astronomy, government, law, location, tv, computer, business, \
            film, medicine, military, time, organization, people, sports, \
            soccer, architecture, geography, music, games, royalty, education, \
            award, biology, internet, symbols, book, media_common, visual_art, \
            travel, fictional_universe, aviation, transportation, chemistry, \
            language, finance, automotive, opera, comic_books, basketball, \
            food, interests, theater, religion, measurement_unit, fashion, \
            meteorology, engineering, exhibitions, physics \n\
            ---------------\n\
            You will be given a text excerpt by the user in Turkish and will \
            have to output just the name of the domain you think applies most \
            to the excerpt. Below are examples of message-output pairs that are \
            correct:\n\
            ---------------\n\
            'message': Laborie , Saint Lucia'yı oluşturan 11 idarî şehrinden biridir\
            'output': location\n\
            ---------------\n\
            'message': İnci Küpeli Kız romanı ve bu romandan uyarlanan aynı isimli \
            film Vermeer'in İnci Küpeli Kız tablosundaki modeliyle olan ilişkisini \
            anlatır\
            'output': visual_art\n\
            ---------------\n\
            These examples were correct as they only output the domain most closely \
            associated with the excerpt. Here are two examples NOT to follow:\n\
            'message': Laborie , Saint Lucia'yı oluşturan 11 idarî şehrinden biridir\
            'output': place\n\
            - place is not a domain, even though it is synonymous to location. Do not\
            make up new domain names, use the ones listed\n\
            'message': Süre , Snow Leopard\'ın bağımsız perakende sürümü için lisans \
            Mac OS X v10.5 \" Leopard \" dan kullanıcılar için yükseltmeyi kısıtlıyor\
            'output': computer because Mac OSX Leopard is associated with Mac computers\n\
            - While the domain is correct, we do not want a reasoning, rather just the \
            name of the domain.\n\
            ---------------\n\
            Use the examples given to output existing domain names you most closely \
            associate with the excerpts. Do not add any artifacts such as a new line, \
            just the domain name from the list provided. Do NOT even provide the \
            'output' tag I added, that is for presenting purposes"
    ),
    (
        "human", 
        x),
]

# ai_msg = llm.invoke(messages)

# print(ai_msg.content)