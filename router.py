from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.index import LocalIndex

encoder = HuggingFaceEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

faq = Route(
    name="faq",
    utterances=[
        "What is the return policy of the products?",
        "Do I get discount with the HDFC credit card?",
        "How can I track my order?",
        "What payment methods are accepted?",
        "How long does it take to process a refund?",
        "What happens if I receive a defective product?",
        "What is your policy on damaged or faulty items?",
        "Can I return a broken or damaged product?",
        "What are your shipping charges?",
        "How do I cancel my order?",
        "Do you offer cash on delivery?",
        "What is the warranty on products?",
    ],
)

sql = Route(
    name="sql",
    utterances=[
        "I want to buy nike shoes that have 50% discount.",
        "Are there any shoes under Rs. 3000?",
        "Do you have formal shoes in size 9?",
        "Are there any Puma shoes on sale?",
        "What is the price of puma running shoes?",
        "Show me Nike shoes with rating more than 4",
        "Find shoes with high ratings",
        "Show me top rated products",
        "List shoes under 2000 rupees",
        "Show me discounted Adidas shoes",
        "What are the cheapest running shoes available?",
        "Find me products with more than 500 reviews",
    ],
)

def build_router():
    index = LocalIndex()
    _router = SemanticRouter(
        routes=[faq, sql],
        encoder=encoder,
        index=index,
    )
    _router.add(routes=[faq, sql])
    return _router

if __name__ == "__main__":
    router = build_router()
    print(router("What is your policy on defective product?").name)
    print(router("Show me Nike shoes with rating more than 4").name)
    print(router("Pink Puma shoes in price range 5000 to 10000").name)