import pandas as pd

# Original data
data = {
    "Type": [
        "0-animal", "1-clothes", "2-color", "3-decorates", "4-food",
        "5-object", "6-plants", "7-text", "8-texture", "9-Transportation"
    ],
    "Prompt1": [
        "A photo of a bear riding a bicycle",
        "A portrait of a woman wearing a dress",
        "A bear holding a blue balloon.",
        "A cake decorated with chocolate chips",
        "A painting of a girl holding a banana",
        "A photo of a ball on the table",
        "A drawing of a garden filled with sunflowers",
        "A magazine cover with the headline “STYLE”",
        "A cup with a smooth surface",
        "A drawing of a boy driving a car"
    ],
    "Prompt2": [
        "A photo of a cat riding a bicycle",
        "A portrait of a woman wearing a jacket",
        "A bear holding a red balloon.",
        "A cake decorated with strawberries",
        "A painting of a girl holding an apple",
        "A photo of a book on the table",
        "A drawing of a garden filled with roses",
        "A magazine cover with the headline “FASHION”",
        "A cup with a rough surface",
        "A drawing of a boy driving a truck"
    ],
    "Prompt3": [
        "A photo of a dog riding a bicycle",
        "A portrait of a woman wearing a sweater",
        "A bear holding a yellow balloon.",
        "A cake decorated with blueberries",
        "A painting of a girl holding a grape",
        "A photo of a cup on the table",
        "A drawing of a garden filled with tulips",
        "A magazine cover with the headline “DESIGN”",
        "A cup with a shiny surface",
        "A drawing of a boy driving a motorcycle"
    ],
    "Prompt4": [
        "A photo of a lion riding a bicycle",
        "A portrait of a woman wearing a skirt",
        "A bear holding a black balloon.",
        "A cake decorated with whipped cream",
        "A painting of a girl holding a pineapple",
        "A photo of a lamp on the table",
        "A drawing of a garden filled with daisies",
        "A magazine cover with the headline “TRENDS”",
        "A cup with a matte surface",
        "A drawing of a boy driving a bus"
    ],
    "Prompt5": [
        "A photo of a tiger riding a bicycle",
        "A portrait of a woman wearing a coat",
        "A bear holding a pink balloon.",
        "A cake decorated with sprinkles",
        "A painting of a girl holding a watermelon",
        "A photo of a phone on the table",
        "A drawing of a garden filled with lavender",
        "A magazine cover with the headline “BEAUTY”",
        "A cup with a soft surface",
        "A drawing of a boy driving a bicycle"
    ],
    "Prompt6": [
        "A photo of a rabbit riding a bicycle",
        "A portrait of a woman wearing a shirt",
        "A bear holding a white balloon.",
        "A cake decorated with almonds",
        "A painting of a girl holding a mango",
        "A photo of a chair on the table",
        "A drawing of a garden filled with lotus",
        "A magazine cover with the headline “LIFESTYLE”",
        "A cup with a hard surface",
        "A drawing of a boy driving a scooter"
    ],
    "Prompt7": [
        "A photo of a panda riding a bicycle",
        "A portrait of a woman wearing a scarf",
        "A bear holding a green balloon.",
        "A cake decorated with cherries",
        "A painting of a girl holding a strawberry",
        "A photo of a table on the table",
        "A drawing of a garden filled with orchids",
        "A magazine cover with the headline “MODERN”",
        "A cup with a sticky surface",
        "A drawing of a boy driving a train"
    ],
    "Prompt8": [
        "A photo of a fox riding a bicycle",
        "A portrait of a woman wearing a hoodie",
        "A bear holding a purple balloon.",
        "A cake decorated with mint leaves",
        "A painting of a girl holding a peach",
        "A photo of a pen on the table",
        "A drawing of a garden filled with peonies",
        "A magazine cover with the headline “ELEGANCE”",
        "A cup with a wet surface",
        "A drawing of a boy driving a submarine"
    ],
    "Prompt9": [
        "A photo of a wolf riding a bicycle",
        "A portrait of a woman wearing a blouse",
        "A bear holding an orange balloon.",
        "A cake decorated with macarons",
        "A painting of a girl holding a papaya",
        "A photo of a key on the table",
        "A drawing of a garden filled with violets",
        "A magazine cover with the headline “GLAMOUR”",
        "A cup with a dry surface",
        "A drawing of a boy driving a helicopter"
    ],
    "Prompt10": [
        "A photo of a monkey riding a bicycle",
        "A portrait of a woman wearing a suit",
        "A bear holding a gray balloon.",
        "A cake decorated with caramel drizzle",
        "A painting of a girl holding a kiwi",
        "A photo of a bag on the table",
        "A drawing of a garden filled with carnations",
        "A magazine cover with the headline “FEATURES”",
        "A cup with a grainy surface",
        "A drawing of a boy driving a jet"
    ],
    "Prompt11": [
        "A photo of an elephant riding a bicycle",
        "A portrait of a woman wearing a vest",
        "A bear holding a brown balloon.",
        "A cake decorated with raspberries",
        "A painting of a girl holding an orange",
        "A photo of a clock on the table",
        "A drawing of a garden filled with hydrangeas",
        "A magazine cover with the headline “INSPIRE”",
        "A cup with a fuzzy surface",
        "A drawing of a boy driving a spaceship"
    ]
}


df = pd.DataFrame(data)

# Desired new order
new_order = [
    "0-animal", "1-clothes", "2-color", "3-decorates", "4-food",
    "5-object", "6-plants", "7-text", "8-texture", "9-Transportation"
]

# Reorder rows
df = df.set_index("Type").loc[new_order].reset_index()

# Save CSV
df.to_csv("prompts.csv", index=False)
print("Saved reordered prompts.csv")
