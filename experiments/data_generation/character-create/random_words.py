import random
def get_random_words(n):
    random_list = [
        "apple", "banana", "chair", "table", "pen", "John", "Sarah", "Emily", "plumber", "teacher", 
        "nurse", "cooking", "reading", "knitting", "shirt", "pants", "dress", "skirt", "kind", 
        "honest", "tall", "weather", "politics", "news", "coffee", "cup", "plate", "Mike", "Emma", 
        "gardener", "baker", "hiking", "painting", "hat", "scarf", "gloves", "thoughtful", "funny", 
        "music", "sports", "health", "fork", "spoon", "Matt", "Lisa", "cashier", "janitor", "swimming", 
        "writing", "shoes", "sandals", "boots", "adventurous", "brave", "family", "movies", "food", 
        "clock", "lamp", "Jane", "Brian", "barista", "librarian", "jogging", "gardening", "coat", 
        "jacket", "athletic", "patient", "work", "travel", "hobbies", "water", "jug", "Tom", "Anne", 
        "firefighter", "mailman", "yoga", "drawing", "jeans", "suit", "generous", "loud", "technology", 
        "culture", "education", "knife", "hammer", "Rachel", "Steve", "mechanic", "hairdresser", 
        "fishing", "dancing", "blouse", "leggings", "ambitious", "quiet", "holidays", "finance", 
        "pets", "bed", "desk", "Chris", "Megan", "receptionist", "cook", "camping", "chess", "t-shirt", 
        "sweater", "humble", "outspoken", "books", "lifestyle", "shopping", "mirror", "comb", "Katie", 
        "Dan", "security_guard", "florist", "photography", "scrapbooking", "shorts", "uniform", 
        "optimistic", "curious", "relationships", "wellness", "wallet", "keys", "Laura", "Tim", 
        "truck_driver", "babysitter", "birdwatching", "calligraphy", "overalls", "tank_top", "stubborn", 
        "polite", "current_events", "social_media", "door", "pillow", "Alan", "Wendy", "taxi_driver", 
        "housekeeper", "golfing", "singing", "hoodie", "vest", "confident", "shy", "religion", "career", 
        "laptop", "phone",
        "orange", "grape", "couch", "desk", "pencil", "Robert", "Olivia", "Sophia", "electrician", "professor", 
        "doctor", "baking", "writing", "sewing", "blouse", "jeans", "jumpsuit", "tie", 
        "truthful", "loyal", "short", "climate", "government", "headlines", "tea", "mug", "bowl", "Daniel", 
        "Ava", "landscaper", "chef", "mountaineering", "sculpture", "cap", "shawl", "mittens", "considerate", 
        "witty", "art", "athletics", "well-being", "knife", "fork", "Mark", "Linda", "waiter", "custodian", 
        "diving", "journaling", "sneakers", "slippers", "shoes", "exploratory", "fearless", "relatives", 
        "cinema", "cuisine", "watch", "lantern", "Mary", "Joseph", "bartender", "archivist", "running", 
        "horticulture", "jacket", "coat", "muscular", "tolerant", "employment", "journey", "pastimes", 
        "fluid", "pitcher", "Nancy", "Robert", "paramedic", "mail carrier", "meditation", "painting", 
        "trousers", "tuxedo", "altruistic", "soft-spoken", "novels", "liveliness", "boutiques", "reflection", 
        "brush", "Elizabeth", "William", "technician", "stylist", "angling", "ballroom", "blouse", "leggings", 
        "aspiring", "reserved", "vacations", "economics", "pets", "couch", "lamp", "Liam", "Emma", "chauffeur", 
        "nanny", "ornithology", "penmanship", "coveralls", "camisole", "stubborn", "courteous", "current affairs", 
        "online platforms", "gate", "cushion", "Andrew", "Michelle", "bus driver", "housemaid", "golf", "vocalizing", 
        "sweatshirt", "waistcoat", "self-assured", "introverted", "spirituality", "occupation", "computer", "smartphone"
        "pear", "kiwi", "sofa", "bench", "marker", "William", "Sophie", "Grace", "mechanical_engineer", "instructor", 
        "physician", "grilling", "typing", "quilting", "blazer", "shorts", "gown", "bowtie", 
        "compassionate", "trustworthy", "average_height", "climate_conditions", "policies", "headlines", "cappuccino", 
        "saucer", "dish", "Christopher", "Oliver", "landscaping_artist", "pastry_chef", "exploration", "pottery", 
        "beanie", "poncho", "gloves", "attentive", "humorous", "symphony", "athletics", "wellness", "spatula", 
        "knife", "fork", "Rachel", "Julia", "waitress", "custodial_worker", "sailing", "note-taking", "loafers", 
        "flip-flops", "boots", "bold", "daring", "relatives", "theater", "cuisine", "watch", "lamp", "Gary", 
        "Ella", "mixologist", "archaeologist", "sprint", "floristry", "raincoat", "parka", "muscular", "patient", 
        "employment", "exploration", "pastimes", "liquid", "pitcher", "Patricia", "John", "paramedic", "mail_carrier", 
        "pilates", "sketching", "slacks", "formalwear", "charitable", "reserved", "fiction", "enthusiasm", "boutiques", 
        "reflection", "canvas", "Victoria", "Charles", "technological_specialist", "barber", "angling", "tap_dancing", 
        "blouse", "leggings", "aspiration", "reticent", "vacation", "finance", "animals", "sofa", "chandelier", 
        "Noah", "Sophia", "pilot", "housekeeper", "surfing", "singing", "hood", "waistcoat", "self-reliant", 
        "introvert", "faith", "vocation", "personal_computer", "cellphone",
        "grapefruit", "blueberry", "couch", "bench", "crayon", "Alexander", "Isabella", "Sophie", "software_engineer", "lecturer", 
        "surgeon", "baking", "typing", "origami", "blouse", "jeans", "robe", "necktie", 
        "empathetic", "dependable", "average_height", "climate_change", "government", "headlines", "espresso", 
        "teapot", "tray", "Daniel", "Ella", "landscaper", "sous_chef", "trekking", "sculpture", 
        "beanie", "shawl", "mittens", "considerate", "witty", "symphony", "athletics", "well-being", 
        "spoon", "knife", "Michael", "Olivia", "waiter", "caretaker", "snorkeling", "journaling", 
        "sneakers", "sandals", "boots", "adventurous", "fearless", "relatives", "film", "cuisine", 
        "watch", "lamp", "Matthew", "Ava", "mixologist", "librarian", "running", "gardening", 
        "raincoat", "jacket", "muscular", "tolerant", "employment", "exploration", "pastimes", 
        "liquid", "jug", "Natalie", "William", "paramedic", "postal_worker", "meditation", "painting", 
        "trousers", "tuxedo", "altruistic", "soft-spoken", "novels", "liveliness", "boutiques", 
        "reflection", "canvas", "Oliver", "Lily", "technician", "hair_stylist", "fishing", "ballroom_dancing", 
        "blouse", "leggings", "aspiring", "reserved", "vacations", "finance", "pets", "couch", 
        "lantern", "Sophia", "Elijah", "pilot", "housemaid", "swimming", "singing", "sweater", 
        "vest", "self-confident", "introverted", "spirituality", "profession", "computer", "mobile_phone",
        "watermelon", "strawberry", "sofa", "bench", "marker", "James", "Sophia", "Emma", "data_scientist", "professor", 
        "dentist", "grilling", "coding", "pottery", "blouse", "shorts", "gown", "bowtie", 
        "compassionate", "reliable", "average_height", "climate_change", "government", "headlines", "latte", 
        "teacup", "dish", "William", "Olivia", "landscaper", "pastry_chef", "exploration", "ceramics", 
        "beanie", "poncho", "gloves", "attentive", "amusing", "concert", "athletics", "well-being", 
        "spatula", "knife", "Sophia", "Liam", "waitress", "custodian", "sailing", "sketching", 
        "loafers", "flip-flops", "boots", "bold", "adventurous", "relatives", "theater", "cuisine", 
        "watch", "lamp", "Ethan", "Ava", "bartender", "historian", "sprint", "horticulture", 
        "raincoat", "jacket", "muscular", "patient", "employment", "exploration", "pastimes", 
        "liquid", "jug", "Isabella", "Michael", "paramedic", "postal_worker", "meditation", "painting", 
        "trousers", "tuxedo", "altruistic", "soft-spoken", "fiction", "liveliness", "boutiques", 
        "reflection", "canvas", "Mason", "Sophie", "technician", "barber", "angling", "dance", 
        "blouse", "leggings", "aspiring", "reserved", "vacations", "finance", "pets", "couch", 
        "lantern", "Ava", "Liam", "pilot", "housekeeper", "surfing", "singing", "hoodie", 
        "waistcoat", "self-assured", "introverted", "faith", "profession", "computer", "smartphone"
    ]

    return random.sample(random_list, n)
