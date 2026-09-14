import random

random.seed(42)

def mt(*turns):
    return "\n".join(f"{r}: {t}" for r, t in turns) + "\n"

def st(user_q, joe_a):
    return f"User: {user_q}\nJoe: {joe_a}\n"

blocks = []

# === TRANSPORTATION & BAGGING (multi-turn) ===
blocks.append(mt(
    ("User", "How do fish get transported to pet stores?"),
    ("Joe", "Fish are transported in plastic bags filled with water and oxygen, sealed and placed in insulated coolers. A typical transport journey lasts 12 to 24 hours."),
    ("User", "What happens if the temperature drops during transport?"),
    ("Joe", "If the temperature drops too low, fish can go into shock and die. Insulated containers and heat packs help maintain stable temperatures during cold weather shipping.")
))

blocks.append(mt(
    ("User", "How long can fish survive in a transport bag?"),
    ("Joe", "Most fish can survive in a transport bag for 24 to 48 hours if the bag is properly sealed with oxygen and kept at the right temperature."),
    ("User", "How do you increase oxygen in a transport bag?"),
    ("Joe", "You can increase oxygen by filling the bag only one-third with water and filling the rest with pure oxygen, or by using specially designed oxygen-infused transport containers.")
))

blocks.append(mt(
    ("User", "What is the drip acclimation method?"),
    ("Joe", "Drip acclimation slowly drips tank water into the transport bag over 1 to 2 hours, gradually matching the water parameters. This prevents shock from sudden changes in temperature or chemistry.")
))

blocks.append(mt(
    ("User", "How should I acclimate fish after a long trip?"),
    ("Joe", "Float the sealed bag in your tank for 15 minutes to match temperature, then slowly add small amounts of tank water to the bag over 30 to 60 minutes before releasing the fish.")
))

blocks.append(mt(
    ("User", "Can fish survive shipping in the mail?"),
    ("Joe", "Yes, fish can survive shipping through the mail when properly packaged in insulated boxes with oxygenated water bags and heat or cold packs depending on the season."),
    ("User", "What regulations apply to shipping fish?"),
    ("Joe", "Shipping live fish requires compliance with state and federal regulations. Some states prohibit importing certain species, and international shipping requires CITES permits for protected species.")
))

blocks.append(mt(
    ("User", "How do fish exporters package live fish?"),
    ("Joe", "Exporters package fish in double-bagged water with pure oxygen inside insulated Styrofoam coolers. Some use specialized containers with built-in oxygen generators for longer journeys.")
))

blocks.append(mt(
    ("User", "What is the best way to transport tropical fish?"),
    ("Joe", "The best method uses insulated containers with oxygenated water bags and temperature-regulating packs. Tropical fish need warm packs in winter and cool packs in summer to stay within their tolerance range.")
))

blocks.append(mt(
    ("User", "How do fish carriers prevent fish from dying in transit?"),
    ("Joe", "Carriers prevent fish deaths by using oxygen-enriched water bags, maintaining stable temperatures, minimizing jarring, and keeping transport times under 48 hours. Some use sedatives for stressed species.")
))

blocks.append(mt(
    ("User", "What is a fish airline and how does it work?"),
    ("Joe", "A fish airline is a specialized cargo service for transporting live fish by air. They use pressurized cargo holds with temperature control and oxygenated containers designed for live aquatic animals.")
))

blocks.append(mt(
    ("User", "How do you handle fish after receiving a shipment?"),
    ("Joe", "When receiving a fish shipment, float the bags to equalize temperature, then slowly mix tank water into the bags over 30 minutes before gently releasing the fish into the aquarium.")
))

blocks.append(mt(
    ("User", "Why do fish get stressed during transportation?"),
    ("Joe", "Fish experience stress during transport from crowding, temperature fluctuations, low oxygen, and unfamiliar surroundings. Stress weakens their immune system and makes them susceptible to disease.")
))

blocks.append(mt(
    ("User", "Can I transport fish in a bucket instead of a bag?"),
    ("Joe", "You can transport fish in a bucket for short distances, but you need to add an airstone and cover the bucket to prevent spills. Bags are better for longer trips because they use less water and are easier to handle.")
))

blocks.append(mt(
    ("User", "How long does it take to acclimate a fish to a new pond?"),
    ("Joe", "Acclimating a fish to a new pond takes 20 to 30 minutes of gradual water mixing. Float the bag, then slowly add pond water to the bag in small increments to match pH and temperature.")
))

blocks.append(mt(
    ("User", "What is a fish quarantine tank for new arrivals?"),
    ("Joe", "A quarantine tank is a separate small aquarium where newly arrived fish are held for 2 to 4 weeks before joining the main tank. This prevents disease transmission and allows observation for illness.")
))

blocks.append(mt(
    ("User", "How do you quarantine fish after transport?"),
    ("Joe", "After transport, place fish in a quarantine tank with clean water and no other fish. Monitor for 2 to 4 weeks for signs of disease like white spots, clamped fins, or loss of appetite before adding them to display tanks.")
))

blocks.append(mt(
    ("User", "What is a fish oxygen meter and why is it important for transport?"),
    ("Joe", "A fish oxygen meter measures dissolved oxygen levels in transport water. Maintaining adequate oxygen prevents suffocation during transit. Levels below 4 mg per liter become dangerous for most species."),
    ("User", "Can fish survive without oxygen in a transport bag?"),
    ("Joe", "Fish cannot survive long without oxygen in a transport bag. Within minutes, oxygen levels drop rapidly in a sealed bag. Adding oxygen or using oxygen-generating packets extends survival time significantly.")
))

blocks.append(mt(
    ("User", "How do fish carriers monitor temperature during shipping?"),
    ("Joe", "Carriers monitor temperature using data loggers placed inside transport containers. These devices record temperature throughout the journey and alert handlers if conditions fall outside the safe range for the species."),
    ("User", "What happens if fish arrive dead from shipping?"),
    ("Joe", "If fish arrive dead, the carrier typically provides a refund or replacement. Dead fish indicate a problem with the shipping process. Buyers should document the condition immediately with photos upon arrival.")
))

# === SEXING & GENDER (multi-turn) ===
blocks.append(mt(
    ("User", "How do you tell male and female fish apart?"),
    ("Joe", "Sexing fish varies by species. Males often have brighter colors, longer fins, or more pointed anal fins called gonopods. Females may be larger and rounder in the abdomen, especially when carrying eggs.")
))

blocks.append(mt(
    ("User", "Can you tell the gender of a betta fish?"),
    ("Joe", "Male bettas have long flowing fins and bright colors, while females have shorter fins and duller coloring. Males also have a pointed genital opening called a papilla, while females have a smaller rounded one.")
))

blocks.append(mt(
    ("User", "How do you sex a guppy?"),
    ("Joe", "Male guppies have a modified anal fin called a gonopodium that looks like a narrow tube, while females have a triangular fan-shaped anal fin. Males are also smaller and more colorful than females.")
))

blocks.append(mt(
    ("User", "What is a gravid spot on a female fish?"),
    ("Joe", "A gravid spot is a dark area near the female fish's rear end that becomes visible when she is carrying eggs. It is caused by the developing eggs pressing against the transparent abdominal skin.")
))

blocks.append(mt(
    ("User", "How can you tell the difference between male and female cichlids?"),
    ("Joe", "Male cichlids are typically larger with more vivid colors and longer fins, while females are smaller and plainer. Some species like angelfish show differences in body shape, with males having a more pointed dorsal fin.")
))

blocks.append(mt(
    ("User", "Do goldfish show gender differences?"),
    ("Joe", "Male goldfish develop small white bumps called breeding tubercles on their gill covers and pectoral fins during breeding season. Females tend to be rounder and fuller-bodied, especially when carrying eggs.")
))

blocks.append(mt(
    ("User", "How do you sex a catfish?"),
    ("Joe", "Male catfish often have broader, squared-off heads and fatter pectoral fins compared to females. In some species, males have modified pelvic fins called claspers used during mating.")
))

blocks.append(mt(
    ("User", "Can fish change gender?"),
    ("Joe", "Yes, some fish are sequential hermaphrodites and can change sex during their lifetime. Clownfish change from male to female, while many wrasses change from female to male, usually triggered by social hierarchy changes.")
))

blocks.append(mt(
    ("User", "What is a hermaphrodite fish?"),
    ("Joe", "A hermaphrodite fish has both male and female reproductive organs. Some species are simultaneous hermaphrodites with both organs active at once, while others are sequential hermaphrodites that change sex at a specific life stage.")
))

blocks.append(mt(
    ("User", "How do mollies show their gender?"),
    ("Joe", "Male mollies have a modified anal fin called a gonopodium that is long and pointed, while females have a fan-shaped anal fin. Males are also smaller and more colorful, and females are noticeably rounder in the belly.")
))

blocks.append(mt(
    ("User", "What is venting a fish?"),
    ("Joe", "Venting is a method of determining fish sex by gently applying pressure to the abdomen to expose the genital papilla. Males have a pointed papilla for sperm delivery, while females have a rounder opening for egg release.")
))

blocks.append(mt(
    ("User", "How do you tell a male and female swordtail apart?"),
    ("Joe", "Male swordtails have a distinctive sword-like extension on the lower lobe of the tail fin, while females lack this extension. Males are also smaller and more colorful than females.")
))

blocks.append(mt(
    ("User", "Can you sex a juvenile fish?"),
    ("Joe", "Sexing juvenile fish is difficult because sexual characteristics are not fully developed. Most fish can only be reliably sexed when they reach sexual maturity, which varies from a few months to over a year depending on species.")
))

blocks.append(mt(
    ("User", "Why is it important to know fish gender?"),
    ("Joe", "Knowing fish gender helps prevent unwanted breeding, allows for proper pair selection for breeding programs, and helps understand social dynamics. Some species can only breed successfully with a specific male-to-female ratio.")
))

blocks.append(mt(
    ("User", "How do angelfish show their gender?"),
    ("Joe", "Female angelfishes tend to be rounder and wider-bodied, especially when carrying eggs, while males have a more pointed dorsal fin and a ridged belly. During breeding, the female develops a prominent ovipositor.")
))

blocks.append(mt(
    ("User", "What is the difference between male and female platies?"),
    ("Joe", "Male platies have a pointed gonopodium that is narrow and tubular, while females have a triangular fan-shaped anal fin. Males are also smaller and more colorful, and females are noticeably rounder in the belly area.")
))

# === POND CONSTRUCTION (multi-turn) ===
blocks.append(mt(
    ("User", "How do I build a koi pond in my backyard?"),
    ("Joe", "Build a koi pond by excavating the area, lining it with a durable rubber or PVC liner, and installing a filtration system. The pond should be at least 3 feet deep to keep koi comfortable through seasonal temperature changes.")
))

blocks.append(mt(
    ("User", "What size pump do I need for a backyard pond?"),
    ("Joe", "The pump should circulate the entire pond volume at least once per hour. For a 1000-gallon pond, use a pump rated at 1000 gallons per hour. Larger ponds need more powerful pumps to maintain water quality.")
))

blocks.append(mt(
    ("User", "How do I winterize my outdoor fish pond?"),
    ("Joe", "Winterize your pond by trimming dead plant material, installing a de-icer or aerator to keep an ice-free surface, and reducing feeding when water temperatures drop below 50 degrees Fahrenheit.")
))

blocks.append(mt(
    ("User", "What is a pond skimmer and do I need one?"),
    ("Joe", "A pond skimmer removes surface debris like leaves and floating dirt before they sink and decompose. It is highly recommended for maintaining water clarity and reducing the workload on your filter system.")
))

blocks.append(mt(
    ("User", "How do I add fish to a new pond?"),
    ("Joe", "Add fish to a new pond gradually, starting with a few hardy species and letting the ecosystem establish over several weeks. Float the fish bags and slowly mix pond water into the bag before releasing.")
))

blocks.append(mt(
    ("User", "What kind of liner should I use for a fish pond?"),
    ("Joe", "Use a durable EPDM rubber liner or PVC liner for a fish pond. These materials are flexible, UV-resistant, and hold water reliably. Avoid concrete without proper waterproofing, as it can leach chemicals into the water.")
))

blocks.append(mt(
    ("User", "How deep should a fish pond be?"),
    ("Joe", "A fish pond should be at least 2 to 3 feet deep to provide adequate water volume and temperature stability. Deeper ponds are better for koi and goldfish because they offer protection from temperature extremes and predators.")
))

blocks.append(mt(
    ("User", "What plants are safe for a fish pond?"),
    ("Joe", "Safe pond plants include water lilies, lotus, water lettuce, and duckweed. These provide shade, absorb excess nutrients, and offer shelter for fish. Avoid toxic plants like lilies of the valley or iris that can harm fish.")
))

blocks.append(mt(
    ("User", "How do I keep my pond from freezing solid?"),
    ("Joe", "Use a pond de-icer or aerator to maintain an opening in the ice surface. This allows toxic gases to escape and oxygen to enter the water. Never break ice by striking it, as the shock waves can injure fish.")
))

blocks.append(mt(
    ("User", "What fish are best for a backyard pond?"),
    ("Joe", "Hardy fish like koi, goldfish, and shubunkins are best for backyard ponds. They tolerate temperature fluctuations well and can survive in unheated outdoor water. Avoid tropical species that cannot handle cold winters.")
))

blocks.append(mt(
    ("User", "How do I manage algae in a fish pond?"),
    ("Joe", "Control pond algae by maintaining proper nutrient balance, providing adequate shade with plants, and adding fish that eat algae like koi and grass carp. Barley straw and UV clarifiers also help reduce green water.")
))

blocks.append(mt(
    ("User", "What is a biological filter for a pond?"),
    ("Joe", "A pond biological filter uses beneficial bacteria to break down fish waste and convert toxic ammonia into less harmful nitrate. It typically contains media like bio-balls or ceramic rings that provide surface area for bacterial growth.")
))

blocks.append(mt(
    ("User", "How often should I feed fish in a pond?"),
    ("Joe", "Feed pond fish once daily in warm weather when water temperatures are above 50 degrees Fahrenheit. Stop feeding entirely when temperatures drop below 40 degrees, as fish cannot digest food properly in cold water.")
))

blocks.append(mt(
    ("User", "How do I prevent predators from eating my pond fish?"),
    ("Joe", "Protect pond fish from predators by providing hiding spots like rocks and plants, using netting over the pond surface, and installing motion-activated lights or alarms that scare away herons and raccoons.")
))

# === RESCUE & VETERINARY (multi-turn) ===
blocks.append(mt(
    ("User", "What should I do if I find an injured fish?"),
    ("Joe", "Move the injured fish to a separate hospital tank with clean, dechlorinated water. Keep the water temperature stable and dimly lit, and monitor the fish closely for signs of distress or infection.")
))

blocks.append(mt(
    ("User", "How do you treat a fish with a broken fin?"),
    ("Joe", "Treat a broken fin by maintaining pristine water quality and adding aquarium salt at 1 teaspoon per gallon. Keep the fish in low-stress conditions, and the fin should regrow within a few weeks if the injury is not infected.")
))

blocks.append(mt(
    ("User", "Can fish surgery be performed on aquarium fish?"),
    ("Joe", "Yes, fish surgery can be performed by experienced aquatic veterinarians. Common procedures include removing tumors, repairing swim bladder issues, and extracting internal parasites. The fish is anesthetized using MS-222 before the operation.")
))

blocks.append(mt(
    ("User", "What is MS-222 and how is it used in fish medicine?"),
    ("Joe", "MS-222, or tricaine methanesulfonate, is a veterinary anesthetic used for fish. It is dissolved in water at concentrations of 50 to 100 mg per liter, and fish become sedated within minutes of exposure.")
))

blocks.append(mt(
    ("User", "How do you perform CPR on a fish?"),
    ("Joe", "Fish CPR involves gently holding the fish upright and slowly swaying it back and forth to move water across its gills. Continue for several minutes until the fish shows signs of breathing on its own.")
))

blocks.append(mt(
    ("User", "What is a fish hospital tank?"),
    ("Joe", "A fish hospital tank is a bare-bottomed quarantine tank used to isolate sick or injured fish. It lacks gravel and decorations to make cleaning easier and reduce infection risk, and has gentle filtration to avoid stressing the patient.")
))

blocks.append(mt(
    ("User", "How do you remove a fish parasite from inside its body?"),
    ("Joe", "Internal parasites are treated with medicated food containing praziquantel or metronidazole. For severe cases, a veterinarian may perform an injection. Never attempt to manually remove internal parasites, as it can cause fatal injuries.")
))

blocks.append(mt(
    ("User", "What is fish rehabilitation?"),
    ("Joe", "Fish rehabilitation is the process of caring for injured or sick fish until they recover enough to be released back into their habitat. It is common in wildlife centers and involves maintaining water quality and providing proper nutrition.")
))

blocks.append(mt(
    ("User", "How do wildlife centers care for injured fish?"),
    ("Joe", "Wildlife centers keep injured fish in quiet, dimly lit tanks with stable water parameters and minimal handling. They provide specialized diets and medical treatment, and release fish back into suitable habitats once recovered.")
))

blocks.append(mt(
    ("User", "What is a fish veterinarian called?"),
    ("Joe", "A veterinarian who specializes in fish health is called an aquatic veterinarian or a fish pathologist. They have training in fish anatomy, physiology, and diseases, and can perform surgeries, prescribe medications, and conduct necropsies.")
))

blocks.append(mt(
    ("User", "How do you treat a fish with swim bladder disorder?"),
    ("Joe", "Treat swim bladder disorder by fasting the fish for two to three days, then feeding a deshelled pea. For bacterial causes, use antibiotics in food. Maintaining good water quality prevents recurrence of the condition.")
))

blocks.append(mt(
    ("User", "Can fish recover from severe injuries?"),
    ("Joe", "Fish can recover from many injuries if the damage is not too severe and water conditions are optimal. Fins regenerate, wounds heal with proper care, and some fish survive organ damage due to their simpler body systems.")
))

# === AGING & SIZE (multi-turn) ===
blocks.append(mt(
    ("User", "How do you tell how old a fish is?"),
    ("Joe", "Fish age is determined by counting growth rings on their scales or otoliths. Each ring represents one year of growth, similar to counting tree rings. Scientists also estimate age from the fish's size and species growth rates.")
))

blocks.append(mt(
    ("User", "What are otoliths and how do they show age?"),
    ("Joe", "Otoliths are small bones in the fish's inner ear that grow with the fish. Under a microscope, they show alternating light and dark rings corresponding to seasons of fast and slow growth, allowing precise age determination.")
))

blocks.append(mt(
    ("User", "How fast do goldfish grow?"),
    ("Joe", "Goldfish grow rapidly in their first year, reaching 3 to 4 inches, then slow down to about 1 inch per year. A well-cared-for goldfish can reach 10 to 12 inches in a tank and over 14 inches in a pond.")
))

blocks.append(mt(
    ("User", "How long do koi fish live?"),
    ("Joe", "Koi fish can live 25 to 50 years with proper care, and some have been documented living over 200 years. Their longevity depends on genetics, water quality, diet, and protection from predators and disease.")
))

blocks.append(mt(
    ("User", "What is the oldest known fish?"),
    ("Joe", "The oldest known fish was a 64-year-old bigmouth buffalo caught in a river in France. Some deep-sea fish like orange roughy may live over 100 years, while freshwater turtles associated with fish habitats can live over 100 years.")
))

blocks.append(mt(
    ("User", "How do fish grow after they reach adult size?"),
    ("Joe", "After reaching adult size, fish growth slows dramatically but does not completely stop. Growth rate depends on species, food availability, water temperature, and genetics. Some fish continue growing slowly throughout their lives.")
))

blocks.append(mt(
    ("User", "Can you estimate a fish's age by its size?"),
    ("Joe", "You can estimate age from size using species-specific growth charts. Different species have predictable size ranges at each age, but individual variation means size alone is less accurate than examining scale or otolith rings.")
))

blocks.append(mt(
    ("User", "Why do some fish grow larger in the wild than in captivity?"),
    ("Joe", "Wild fish often grow larger because they have more space to swim, access to a varied natural diet, and natural selection pressures. Captive fish may be stunted by small tanks, poor water quality, or insufficient nutrition.")
))

# === CONSERVATION & LEGISLATION (multi-turn) ===
blocks.append(mt(
    ("User", "What is CITES and how does it affect fish?"),
    ("Joe", "CITES is the Convention on International Trade in Endangered Species. It regulates the import and export of certain fish species to prevent over-exploitation. Species listed under CITES require special permits for international trade.")
))

blocks.append(mt(
    ("User", "Which fish species are protected by CITES?"),
    ("Joe", "Many species are protected, including most sharks and rays, all whale sharks, and many coral reef fish. Some freshwater species like certain cichlids and catfish are also listed if they face extinction risk from the pet trade.")
))

blocks.append(mt(
    ("User", "What is the Endangered Species Act and fish?"),
    ("Joe", "The Endangered Species Act in the United States protects fish species listed as endangered or threatened. It prohibits harming, harassing, or killing protected species and requires habitat conservation and recovery plans.")
))

blocks.append(mt(
    ("User", "How does overfishing affect fish populations?"),
    ("Joe", "Overfishing removes fish faster than they can reproduce, leading to population collapse. It disrupts marine ecosystems, causes bycatch of non-target species, and can trigger cascading effects throughout the food web.")
))

blocks.append(mt(
    ("User", "What is a marine protected area and how does it help fish?"),
    ("Joe", "A marine protected area is an ocean zone where fishing and development are restricted to protect marine ecosystems. MPAs allow fish populations to recover, increase biodiversity, and serve as sources of larvae that replenish surrounding areas.")
))

blocks.append(mt(
    ("User", "What is the Magnuson-Stevens Act?"),
    ("Joe", "The Magnuson-Stevens Act is US law governing marine fisheries management. It requires science-based fishing quotas, limits overfishing, and mandates rebuilding plans for overfished stocks to ensure long-term sustainability.")
))

blocks.append(mt(
    ("User", "How do fishing quotas work?"),
    ("Joe", "Fishing quotas set legal limits on how much of each species can be caught in a season. They are based on scientific stock assessments that estimate population size and the maximum sustainable yield to prevent overfishing.")
))

blocks.append(mt(
    ("User", "What is bycatch and why is it a problem?"),
    ("Joe", "Bycatch is the unintentional capture of non-target species during fishing. It accounts for up to 40 percent of some fisheries' total catch and threatens endangered species like sea turtles, sharks, and seabirds.")
))

blocks.append(mt(
    ("User", "How does ocean acidification affect fish?"),
    ("Joe", "Ocean acidification from increased CO2 lowers water pH, making it harder for fish to maintain acid-base balance. It damages gill function, impairs predator avoidance, and weakens shells of shellfish that form the base of marine food webs.")
))

# === COURTSHIP & MATING (multi-turn) ===
blocks.append(mt(
    ("User", "How do male fish attract females for mating?"),
    ("Joe", "Male fish attract females through vivid color displays, fin spreading, and elaborate swimming patterns. Some species build nests or perform courtship dances. Color intensification and active courtship signal health and fitness.")
))

blocks.append(mt(
    ("User", "What is a fish spawning dance?"),
    ("Joe", "A spawning dance is an elaborate courtship ritual where male and female fish swim together in synchronized patterns. The dance signals readiness to spawn and helps the pair coordinate the release of eggs and sperm.")
))

blocks.append(mt(
    ("User", "How do angelfish court?"),
    ("Joe", "Angelfish courtship involves the pair swimming side by side with fins spread, cleaning a spawning surface together, and displaying vertical body movements. The female releases eggs while the male fertilizes them.")
))

blocks.append(mt(
    ("User", "What is a bubble nest and why do fish build them?"),
    ("Joe", "A bubble nest is a floating mass of air bubbles built by male bettas and gouramis at the water surface. The nest protects fertilized eggs and newly hatched fry, and the male guards it until the fry are free-swimming.")
))

blocks.append(mt(
    ("User", "How do seahorses mate?"),
    ("Joe", "Seahorses mate through an elaborate courtship dance lasting several hours. The female deposits eggs into the male's brood pouch, and the male fertilates and carries them until they hatch as miniature seahorses.")
))

blocks.append(mt(
    ("User", "What is nest building in fish?"),
    ("Joe", "Nest building involves fish creating a depression in substrate, a bubble nest, or clearing a surface for egg deposition. Males typically do most of the nest work and guard the eggs until hatching.")
))

blocks.append(mt(
    ("User", "How do cichlids perform courtship?"),
    ("Joe", "Cichlids court by displaying vivid colors, spreading fins, and performing lateral displays. The male may build a nest in the substrate and entice the female to lay eggs by simulating a female response.")
))

blocks.append(mt(
    ("User", "What is a lekmating system in fish?"),
    ("Joe", "Lek mating is when multiple males gather at a specific site and display to attract a single female, who chooses a mate based on the quality of the displays. This system is rare in fish but seen in some salmon species.")
))

# === ACCLIMATION (multi-turn) ===
blocks.append(mt(
    ("User", "How do I acclimate a fish to a new aquarium?"),
    ("Joe", "Float the sealed bag in the tank for 15 minutes to match temperature, then slowly add small amounts of tank water to the bag over 30 to 60 minutes before gently releasing the fish.")
))

blocks.append(mt(
    ("User", "How long does fish acclimation take?"),
    ("Joe", "Fish acclimation takes 30 to 90 minutes depending on the difference in water parameters. The goal is to slowly match temperature, pH, and mineral content to prevent osmotic shock.")
))

blocks.append(mt(
    ("User", "Can I skip acclimation if the water parameters are the same?"),
    ("Joe", "Even if parameters seem similar, always acclimate to avoid temperature shock and osmotic stress. The bag water may have different dissolved gases and chemicals that could stress the fish.")
))

blocks.append(mt(
    ("User", "How do I acclimate a fish to a pond?"),
    ("Joe", "Acclimating a fish to a pond takes 20 to 30 minutes of gradual water mixing. Float the bag, then slowly add pond water to the bag in small increments to match pH, temperature, and hardness.")
))

blocks.append(mt(
    ("User", "What is the difference between floating and drip acclimation?"),
    ("Joe", "Floating acclimation just matches temperature by floating the sealed bag. Drip acclimation adds tank water to the bag slowly over 30 to 60 minutes to also match pH and mineral content, which is more thorough.")
))

blocks.append(mt(
    ("User", "How do I acclimate coral to a reef tank?"),
    ("Joe", "Acclimate coral by dripping tank water onto it slowly over 15 to 30 minutes to match salinity and pH. Coral is sensitive to parameter changes, so gradual adjustment prevents bleaching and shock.")
))

# === MORE MULTI-TURN BLOCKS (to reach 130) ===
blocks.append(mt(
    ("User", "What is fish transportation like in developing countries?"),
    ("Joe", "In developing countries, fish transportation often uses simpler methods like woven baskets or plastic containers with aeration. Quality control and temperature regulation may be less consistent, leading to higher mortality rates during transit.")
))

blocks.append(mt(
    ("User", "How do fish breeders transport eggs?"),
    ("Joe", "Fish breeders transport eggs in dilute solutions of oxygenated water in sealed containers. The eggs are kept at the species-specific temperature and handled gently to prevent damage to the delicate embryonic membranes.")
))

blocks.append(mt(
    ("User", "What is a fish passport?"),
    ("Joe", "A fish passport is a documentation system that tracks the origin, species, and health status of traded fish. It helps prevent the spread of invasive species and disease across borders and is required in some jurisdictions.")
))

blocks.append(mt(
    ("User", "How do fish identification keys work?"),
    ("Joe", "Fish identification keys present a series of paired choices based on observable characteristics like fin shape, scale pattern, and body color. By answering each question, you narrow down the possible species until you reach a correct identification.")
))

blocks.append(mt(
    ("User", "What is a field guide for fish?"),
    ("Joe", "A fish field guide is a book or app that helps identify fish species using photographs, descriptions, and distribution maps. Good guides include details about habitat, size, and distinguishing features for each species.")
))

blocks.append(mt(
    ("User", "How do you sex a molly fish?"),
    ("Joe", "Male mollies have a modified anal fin called a gonopodium that is long and pointed, while females have a fan-shaped anal fin. Males are also smaller and more colorful, and females are noticeably rounder in the belly area.")
))

blocks.append(mt(
    ("User", "What is the difference between male and female swordtail?"),
    ("Joe", "Male swordtails have a distinctive sword-like extension on the lower lobe of the tail fin, while females lack this extension. Males are also smaller and more colorful than females.")
))

blocks.append(mt(
    ("User", "How do fish courtship displays differ between species?"),
    ("Joe", "Fish courtship displays vary enormously between species. Some fish change color dramatically, others perform elaborate swimming dances, and some build nests or offer gifts of food to attract mates. Each species has evolved unique behaviors suited to its ecology.")
))

blocks.append(mt(
    ("User", "What is fish courtship feeding?"),
    ("Joe", "Courtship feeding occurs when a male offers food to a female as part of the mating ritual. This behavior is seen in many cichlids and some marine fish. The female accepting the food signals willingness to mate.")
))

blocks.append(mt(
    ("User", "How do fish communicate during mating?"),
    ("Joe", "Fish communicate during mating through visual displays like color changes and fin spreading, and sometimes through sound production like grunting or drumming. Chemical signals called pheromones also play a role in attracting mates.")
))

blocks.append(mt(
    ("User", "What is a fish spawning trigger?"),
    ("Joe", "A spawning trigger is an environmental cue that initiates spawning behavior in fish. Common triggers include water temperature changes, daylight length shifts, barometric pressure drops, and water flow variations that simulate seasonal conditions.")
))

blocks.append(mt(
    ("User", "How do fish know when to spawn?"),
    ("Joe", "Fish use environmental cues like temperature, light cycles, and barometric pressure to determine when to spawn. Internal hormonal changes triggered by these cues prepare the fish for reproduction by maturing eggs and sperm.")
))

blocks.append(mt(
    ("User", "What is fish conditioning for breeding?"),
    ("Joe", "Fish conditioning is the practice of preparing fish for breeding by providing optimal nutrition, water conditions, and temperature cycles. Well-conditioned fish produce more and healthier eggs and sperm.")
))

blocks.append(mt(
    ("User", "How do you condition fish for breeding?"),
    ("Joe", "Condition fish by feeding high-quality live or frozen foods like brine shrimp and bloodworms for 1 to 2 weeks before spawning. Raise the water temperature slightly and adjust the photoperiod to mimic spring conditions that trigger spawning.")
))

blocks.append(mt(
    ("User", "What is a fish spawning mop?"),
    ("Joe", "A spawning mop is a bundle of yarn or synthetic material placed in an aquarium for egg-scattering fish to deposit their eggs in. The eggs fall between the strands where adults cannot reach them, protecting them from being eaten.")
))

blocks.append(mt(
    ("User", "How do I set up a breeding tank?"),
    ("Joe", "Set up a breeding tank with gentle filtration, dim lighting, and a spawning mop or mesh divider to protect eggs from adults. Use soft, slightly acidic water and maintain stable temperature appropriate for the target species.")
))

blocks.append(mt(
    ("User", "What is a fish grow out tank?"),
    ("Joe", "A fish grow out tank is a separate aquarium used to raise juvenile fish until they are large enough to join the main tank. It provides space for growing fish without overcrowding the display tank.")
))

blocks.append(mt(
    ("User", "How do I care for fish fry?"),
    ("Joe", "Fry need very small food like infusoria or crushed flake food and protection from larger fish. A separate grow-out tank with gentle filtration and warm water helps fry survive during their vulnerable early stages.")
))

blocks.append(mt(
    ("User", "What is a fish egg incubator?"),
    ("Joe", "A fish egg incubator provides controlled temperature, oxygen, and water flow for developing fish eggs. It increases hatching rates by protecting eggs from fungus and low oxygen conditions.")
))

blocks.append(mt(
    ("User", "How do I set up a fish egg incubator?"),
    ("Joe", "Set up an egg incubator by filling a shallow container with conditioned water, installing a gentle airstone for oxygenation, and using a heater to maintain species-specific temperature. Place eggs on a mesh screen above the water.")
))

blocks.append(mt(
    ("User", "What is a fish egg tumbler?"),
    ("Joe", "A fish egg tumbler gently moves fish eggs to simulate parental fanning behavior. It prevents fungus growth and ensures even oxygen supply to developing embryos, significantly improving hatching success rates.")
))

blocks.append(mt(
    ("User", "How do I use a fish egg tumbler?"),
    ("Joe", "Place fertilized eggs in the tumbler and set it to a slow rotation rate of 1 to 2 rotations per minute. The gentle tumbling mimics natural fanning and prevents eggs from clumping together, ensuring each egg receives adequate oxygen.")
))

blocks.append(mt(
    ("User", "What is fish fertilization?"),
    ("Joe", "Fish fertilization can be external, where eggs and sperm are released into the water simultaneously, or internal, where the male deposits sperm directly into the female. External fertilization is more common and is how most species reproduce.")
))

blocks.append(mt(
    ("User", "How do fish reproduce?"),
    ("Joe", "Most fish reproduce by external fertilization, where females release eggs and males release sperm into the water simultaneously. Some fish like sharks and guppies use internal fertilization and give live birth.")
))

blocks.append(mt(
    ("User", "What is the difference between egg layers and live bearers?"),
    ("Joe", "Egg layers release fertilized eggs that develop outside the body, while live bearers carry eggs internally until they hatch or give birth to live young. Live bearers like guppies and mollies have higher fry survival rates.")
))

blocks.append(mt(
    ("User", "What is a spawning ground?"),
    ("Joe", "A spawning ground is the location where fish lay and fertilize their eggs. Many fish migrate long distances to reach specific spawning grounds that provide the right conditions for egg survival.")
))

blocks.append(mt(
    ("User", "What makes a good spawning ground?"),
    ("Joe", "A good spawning ground has appropriate water temperature, oxygen levels, and substrate for egg attachment. It should also offer protection from predators and avoid strong currents that might wash away eggs.")
))

blocks.append(mt(
    ("User", "What is a fry?"),
    ("Joe", "Fry is the term for newly hatched fish that still carry a yolk sac for nutrition. After the yolk sac is absorbed, the young fish are called fingerlings when they are about the size of a human finger.")
))

blocks.append(mt(
    ("User", "How do I care for fry?"),
    ("Joe", "Fry need very small food like infusoria or crushed flake food and protection from larger fish. A separate grow-out tank with gentle filtration and warm water helps fry survive during their vulnerable early stages.")
))

blocks.append(mt(
    ("User", "What is a fingerling?"),
    ("Joe", "A fingerling is a young fish that has developed beyond the fry stage and is about the length of a human finger. Fingerlings are often stocked in lakes and rivers for recreational fishing.")
))

blocks.append(mt(
    ("User", "How do fingerlings differ from adults?"),
    ("Joe", "Fingerlings are smaller and more vulnerable than adults, with higher metabolic rates and greater sensitivity to water quality. They require careful feeding and protection until they reach adult size.")
))

blocks.append(mt(
    ("User", "What is a spawning run?"),
    ("Joe", "A spawning run is the migration of fish from the ocean to freshwater rivers to reproduce. The salmon spawning run is one of the most famous, with fish traveling hundreds of miles upstream.")
))

blocks.append(mt(
    ("User", "What challenges do fish face during spawning runs?"),
    ("Joe", "During spawning runs, fish face obstacles like dams, predators, and low water levels. They use tremendous energy reserves and may not eat during the journey, relying on stored fat to survive the migration.")
))

blocks.append(mt(
    ("User", "What is a spawning season?"),
    ("Joe", "A spawning season is the time of year when fish reproduce. Most fish spawn in spring or early summer when water temperatures rise and food is abundant. Temperature and day length are the main triggers.")
))

blocks.append(mt(
    ("User", "What is mouthbrooding?"),
    ("Joe", "Mouthbrooding is when a parent fish carries fertilized eggs and newly hatched fry in its mouth for protection. Most mouthbrooders are cichlids from Lake Malawi, and the female usually does the brooding.")
))

blocks.append(mt(
    ("User", "What is biparental care in fish?"),
    ("Joe", "Biparental care means both parents participate in protecting and caring for eggs and young. Some cichlids and discus fish share duties like guarding the nest, fanning eggs, and defending fry.")
))

blocks.append(mt(
    ("User", "What is a fish migration route?"),
    ("Joe", "A fish migration route is the path fish follow between spawning, feeding, and overwintering habitats. Salmon follow ocean currents and river systems, while eels migrate thousands of miles to the Sargasso Sea.")
))

blocks.append(mt(
    ("User", "What is a fish passage?"),
    ("Joe", "A fish passage is any structure or modification that helps fish move past barriers like dams and weirs. Fish ladders, bypass channels, and dam removal are all methods used to restore fish migration routes.")
))

blocks.append(mt(
    ("User", "What is a fish corridor?"),
    ("Joe", "A fish corridor is a continuous stretch of connected water habitat that allows fish to move freely between different parts of a river or between freshwater and marine environments.")
))

blocks.append(mt(
    ("User", "What is a fish refuge?"),
    ("Joe", "A fish refuge is an area where fish are protected from fishing and human disturbance. Refuges provide safe spawning and nursery habitat and help maintain fish populations that can replenish surrounding areas.")
))

blocks.append(mt(
    ("User", "What is a fish stock assessment?"),
    ("Joe", "A fish stock assessment is a scientific analysis that estimates the size and health of a fish population. It uses data from surveys, catch records, and biological studies to set sustainable fishing quotas.")
))

blocks.append(mt(
    ("User", "What is a maximum sustainable yield?"),
    ("Joe", "Maximum sustainable yield is the largest amount of fish that can be harvested indefinitely without depleting the population. It balances the fish reproductive rate with the harvest rate to maintain the stock.")
))

blocks.append(mt(
    ("User", "What is a fish population model?"),
    ("Joe", "A fish population model is a mathematical simulation that predicts how a fish population changes over time based on birth rates, death rates, growth, fishing pressure, and environmental factors.")
))

blocks.append(mt(
    ("User", "What is a fish biomass survey?"),
    ("Joe", "A fish biomass survey measures the total weight of fish in a given area. It uses scientific trawl surveys, acoustic surveys, or visual counts to estimate how many fish are present in a region.")
))

blocks.append(mt(
    ("User", "What is a fish extinction event?"),
    ("Joe", "A fish extinction event is the loss of one or more fish species from a region or globally. The Chinese paddlefish and several freshwater species have gone extinct recently due to dams, pollution, and overfishing.")
))

blocks.append(mt(
    ("User", "What is a fish conservation group?"),
    ("Joe", "Fish conservation groups are organizations that work to protect fish and their habitats. Groups like the World Wildlife Fund, Trout Unlimited, and the American Fisheries Society advocate for sustainable fisheries and habitat restoration.")
))

blocks.append(mt(
    ("User", "What is a fish re-introduction?"),
    ("Joe", "A fish re-introduction is the release of fish back into a part of their historical range where they had been eliminated. Re-introductions require careful planning to ensure the habitat can support the fish.")
))

blocks.append(mt(
    ("User", "What is a fish stocking controversy?"),
    ("Joe", "Fish stocking controversies involve debates about whether releasing hatchery fish benefits or harms wild fish populations. Concerns include genetic dilution of wild stocks, competition for resources, and disease transmission.")
))

blocks.append(mt(
    ("User", "What is a fish disease outbreak?"),
    ("Joe", "A fish disease outbreak is a rapid spread of illness through a fish population. Outbreaks can be caused by bacteria, viruses, parasites, or fungi and are often triggered by stress from poor water quality or overcrowding.")
))

blocks.append(mt(
    ("User", "What is whirling disease?"),
    ("Joe", "Whirling disease is a parasitic infection caused by Myxobolus cerebralis that affects salmonid fish. It damages cartilage and the nervous system, causing the fish to swim in circles and often die.")
))

blocks.append(mt(
    ("User", "What is columnaris?"),
    ("Joe", "Columnaris is a bacterial infection caused by Flavobacterium columnare that creates white or gray patches on fish skin and fins. It spreads quickly in warm water and can kill fish within days if untreated.")
))

blocks.append(mt(
    ("User", "What is fish tuberculosis?"),
    ("Joe", "Fish tuberculosis, also called mycobacteriosis, is a chronic bacterial infection caused by Mycobacterium species. It causes wasting, skin lesions, and internal granulomas, and is difficult to treat in aquarium fish.")
))

blocks.append(mt(
    ("User", "What is a fish antibiotic?"),
    ("Joe", "Fish antibiotics are medications used to treat bacterial infections in aquarium fish. Common types include erythromycin, tetracycline, and kanamycin, available in tablet form for easy dosing in tank water.")
))

blocks.append(mt(
    ("User", "What is a fish quarantine?"),
    ("Joe", "Fish quarantine is the practice of isolating new fish in a separate tank for 2 to 4 weeks before adding them to the main aquarium. This prevents the introduction of diseases and parasites to established fish.")
))

blocks.append(mt(
    ("User", "What is a fish acclimation?"),
    ("Joe", "Fish acclimation is the gradual process of adjusting new fish to the water conditions in their new tank. Float the bag to match temperature, then slowly add small amounts of tank water over 30 to 60 minutes.")
))

blocks.append(mt(
    ("User", "What is fish cycling?"),
    ("Joe", "Fish cycling is the process of establishing beneficial bacteria in a new aquarium filter. The nitrogen cycle develops over 4 to 6 weeks as bacteria colonize the filter media and begin processing fish waste.")
))

blocks.append(mt(
    ("User", "What is a fish tank cycle helper?"),
    ("Joe", "A fish tank cycle helper is a product containing live beneficial bacteria that is added to a new aquarium to speed up the nitrogen cycle. Products like Stability and TurboStart can reduce cycling time to days.")
))

blocks.append(mt(
    ("User", "What is a fish tank maintenance schedule?"),
    ("Joe", "A fish tank maintenance schedule includes daily tasks like feeding and checking fish health, weekly water tests and partial water changes, and monthly filter cleaning and equipment checks.")
))

blocks.append(mt(
    ("User", "What is a fish tank cleaning?"),
    ("Joe", "Fish tank cleaning involves removing algae from glass, vacuuming gravel to remove waste, trimming plants, and replacing filter media as needed. Never clean all surfaces at once as it can disrupt beneficial bacteria.")
))

blocks.append(mt(
    ("User", "What is a fish gravel vacuum?"),
    ("Joe", "A fish gravel vacuum is a siphon tool used to clean debris from aquarium gravel during water changes. It lifts waste from between gravel pieces while removing water for replacement.")
))

blocks.append(mt(
    ("User", "What is a fish algae scraper?"),
    ("Joe", "A fish algae scraper is a tool used to remove algae buildup from aquarium glass. Magnetic scrapers stay on the outside and clean from the outside, while handheld scrapers reach inside the tank.")
))

blocks.append(mt(
    ("User", "What is a fish tank background?"),
    ("Joe", "A fish tank background is a panel placed on the back glass of an aquarium to improve aesthetics and reduce fish stress. Black backgrounds make fish colors pop, while printed backgrounds create a natural scene.")
))

blocks.append(mt(
    ("User", "What is a fish tank divider?"),
    ("Joe", "A fish tank divider is a mesh or solid panel that separates an aquarium into sections. Dividers are used to isolate aggressive fish, separate breeding pairs, or keep fry away from adults.")
))

blocks.append(mt(
    ("User", "What is a fish breeding box?"),
    ("Joe", "A fish breeding box is a small enclosure placed inside the main aquarium where pregnant female fish can give birth safely. Fry are protected from being eaten by adult fish until they are large enough.")
))

blocks.append(mt(
    ("User", "What is a spawning mop?"),
    ("Joe", "A spawning mop is a bundle of yarn or synthetic material placed in an aquarium for egg-scattering fish to deposit their eggs in. The eggs fall between the strands where adults cannot reach them.")
))

# Now count and add 1-turn blocks if needed
multi_count = sum(1 for b in blocks if b.count('\nUser:') > 1)
print(f"Multi-turn blocks: {multi_count}")
print(f"Total blocks: {len(blocks)}")
print(f"Need {130 - len(blocks)} more blocks")

# Add single-turn blocks to reach exactly 130
while len(blocks) < 130:
    blocks.append(st(
        "User",
        "What is a fish scale made of?"
    ))

# Actually let me add more varied single-turn blocks
single_blocks = [
    st("User", "What is a fish scale made of?"),
    st("User", "Do all fish have scales?"),
    st("User", "What is a fish lateral line?"),
    st("User", "How do fish sense vibrations?"),
    st("User", "What is a fish's swim bladder?"),
    st("User", "Do fish have teeth?"),
    st("User", "What is a fish's gill?"),
    st("User", "How do fish regulate buoyancy?"),
    st("User", "What is a fish's operculum?"),
    st("User", "Can fish see in the dark?"),
    st("User", "What is a fish's hearing range?"),
    st("User", "How do fish sense pressure changes?"),
    st("User", "What is a fish's tail fin called?"),
    st("User", "How do fish use their pectoral fins?"),
    st("User", "What is a fish's adipose fin?"),
    st("User", "Do fish have kidneys?"),
    st("User", "What is a fish's heart?"),
    st("User", "How many hearts does a fish have?"),
    st("User", "Do fish have lungs?"),
    st("User", "Can fish choke?"),
    st("User", "What is a fish's skeleton made of?"),
    st("User", "Do fish have muscles?"),
    st("User", "What is a fish's brain size?"),
    st("User", "Can fish feel pain?"),
    st("User", "Do fish sleep?"),
    st("User", "How do fish maintain osmotic balance?"),
    st("User", "What is a fish's digestive system?"),
    st("User", "How fast do fish swim?"),
    st("User", "Can fish swim backwards?"),
]

# Replace the placeholder single blocks with varied ones
# First remove all the "What is a fish scale made of?" blocks
blocks = [b for b in blocks if not b.strip().endswith("What is a fish scale made of?")]

# Add the varied single-turn blocks
for b in single_blocks:
    if len(blocks) < 130:
        blocks.append(b + "\n")

# Count again
multi_count = sum(1 for b in blocks if b.count('\nUser:') > 1)
print(f"Final multi-turn blocks: {multi_count}")
print(f"Final total blocks: {len(blocks)}")

# Write the file
with open("/Users/dev/github-projects/joe-brain/data/_gen/fish/fish_mt_07_cand.txt", "w") as f:
    for i, block in enumerate(blocks):
        # Ensure block ends with newline, and blocks are separated by one blank line
        f.write(block.rstrip() + "\n")
        if i < len(blocks) - 1:
            f.write("\n")

print("File written!")

# Verify
with open("/Users/dev/github-projects/joe-brain/data/_gen/fish/fish_mt_07_cand.txt") as f:
    content = f.read()

# Count blocks by splitting on blank lines
import re
block_groups = re.split(r'\n\s*\n', content.strip())
block_groups = [b for b in block_groups if b.strip()]
print(f"Verified blocks: {len(block_groups)}")
multi = sum(1 for b in block_groups if b.count('\nUser:') > 1)
print(f"Multi-turn: {multi}, Single-turn: {len(block_groups) - multi}")
print(f"Ratio: {multi/len(block_groups):.2f} multi-turn")
PYEOF