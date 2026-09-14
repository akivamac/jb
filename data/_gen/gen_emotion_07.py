import os

def make_block(turns):
    lines = []
    for role, text in turns:
        lines.append(f"{role}: {text}")
    return "\n".join(lines)

blocks = []

# === Blocks 1-130: emotion domain ===

# 1-10: Anxiety topics
blocks.append(make_block([
    ("User", "I have been worrying constantly about my job security"),
    ("Joe", "Worrying about the future is exhausting. Try focusing on what you can control rather than what might happen."),
    ("User", "what if I lose my job and cannot find another one"),
    ("Joe", "That fear is understandable, but you have skills that are valuable. Start updating your resume now so you feel prepared."),
]))

blocks.append(make_block([
    ("User", "My anxiety keeps me up at night"),
    ("Joe", "Poor sleep makes anxiety worse. Try a calming routine before bed, like reading or gentle stretching."),
    ("User", "how can I quiet my mind when my thoughts race"),
    ("Joe", "Focus on your breathing and name the thoughts as they come without judging them. This practice can loosen their grip over time."),
]))

blocks.append(make_block([
    ("User", "I feel nervous about starting therapy"),
    ("Joe", "Taking that step takes courage. A good therapist will listen without judgment and help you find your own answers."),
    ("User", "what if the therapist does not understand me"),
    ("Joe", "It is okay to feel uncertain about that. You can always try someone else until you find a good fit."),
]))

blocks.append(make_block([
    ("User", "I have social anxiety and dread gatherings"),
    ("Joe", "Social anxiety can make events feel overwhelming. Arriving early to settle in can help you feel more comfortable."),
    ("User", "what if I panic in the middle of a party"),
    ("Joe", "If that happens, step outside for fresh air and take slow breaths. You do not have to stay in the room if you need a break."),
]))

blocks.append(make_block([
    ("User", "I worry about everything all the time"),
    ("Joe", "Chronic worry drains your energy. Try setting aside a specific time each day to write down your worries instead of thinking them all day."),
    ("User", "how do i stop worrying about things that might never happen"),
    ("Joe", "Ask yourself how likely the outcome really is and what you would actually do about it. Most feared outcomes never occur."),
]))

blocks.append(make_block([
    ("User", "I feel a tight knot in my chest when I think about the future"),
    ("Joe", "That physical sensation is a sign of anxiety. Taking slow, deliberate breaths can help ease the tension in your body."),
    ("User", "what if this knot never goes away"),
    ("Joe", "The knot will loosen with practice and self-care. If it persists, a professional can help you find relief."),
]))

blocks.append(make_block([
    ("User", "I get anxious when I have to speak in front of people"),
    ("Joe", "Public speaking anxiety is very common. Practicing in front of a trusted friend can help you build confidence."),
    ("User", "what if I freeze up and forget everything"),
    ("Joe", "Even experienced speakers forget their words sometimes. Having notes or taking a pause is perfectly okay."),
]))

blocks.append(make_block([
    ("User", "My anxiety makes my heart race and my hands shake"),
    ("Joe", "Those physical symptoms are your body's stress response. Grounding techniques can help calm your nervous system."),
    ("User", "what if these symptoms happen at work"),
    ("Joe", "You can excuse yourself to a quiet space and focus on your breathing until the intensity passes."),
]))

blocks.append(make_block([
    ("User", "I avoid checking my email because it triggers my anxiety"),
    ("Joe", "Avoidance can make things worse over time. Try checking email in short, scheduled intervals to reduce the overwhelm."),
    ("User", "what if I still feel anxious even with scheduled checks"),
    ("Joe", "The anxiety will decrease as your brain learns that email is not a threat. Start with very short check-ins."),
]))

blocks.append(make_block([
    ("User", "I feel like my anxiety is controlling my life"),
    ("Joe", "Anxiety can feel all-consuming, but it does not control you. Seeking support and using coping tools can restore your agency."),
    ("User", "what if I have tried everything and nothing works"),
    ("Joe", "Finding the right approach takes time. A combination of professional help and self-care can make a meaningful difference."),
]))

# 11-20: Sadness and grief
blocks.append(make_block([
    ("User", "I am grieving the loss of a close friend"),
    ("Joe", "Grief is a deep and personal process. Allow yourself to feel it without setting a timeline for healing."),
    ("User", "how do I cope when the sadness comes in waves"),
    ("Joe", "Let the waves pass naturally rather than fighting them. Lean on others or write down what you are feeling to release the pain."),
]))

blocks.append(make_block([
    ("User", "I feel so sad about my parents divorce"),
    ("Joe", "Family changes can feel like losing your sense of stability. It is normal to grieve the family structure you once had."),
    ("User", "what if the sadness never goes away"),
    ("Joe", "The sadness will shift and change shape over time even if it does not disappear. Healing is not about forgetting."),
]))

blocks.append(make_block([
    ("User", "I lost a loved one and I do not know how to move on"),
    ("Joe", "Moving on does not mean forgetting. It means finding a way to carry the love and memory with you as you live."),
    ("User", "how do I accept that they are gone"),
    ("Joe", "Acceptance comes slowly and it involves allowing yourself to miss them while also engaging with life."),
]))

blocks.append(make_block([
    ("User", "I feel like nothing matters anymore"),
    ("Joe", "That feeling is a symptom of deep sadness, not the truth. Even small activities can slowly remind you of what brings meaning."),
    ("User", "what if I cannot find anything that matters to me"),
    ("Joe", "Try connecting with one person or doing one small kind act. Meaning often returns through connection rather than searching alone."),
]))

blocks.append(make_block([
    ("User", "I am crying more than usual and I cannot explain why"),
    ("Joe", "Crying can be a release when emotions build up. Let yourself cry and then check in with what you actually need."),
    ("User", "what if crying means something is wrong with me"),
    ("Joe", "Crying is a normal human response, not a sign of weakness. It often signals that you are processing deep feelings."),
]))

blocks.append(make_block([
    ("User", "I feel a deep sadness when I think about how much has changed"),
    ("Joe", "Change can bring profound grief even when it was not expected. Acknowledging that loss is the first step toward healing."),
    ("User", "what if I keep comparing who I am now to who I used to be"),
    ("Joe", "Comparing your past and present self can deepen sadness. Focus on what you are becoming rather than who you were."),
]))

blocks.append(make_block([
    ("User", "I am mourning a version of my life that never happened"),
    ("Joe", "Grieving lost possibilities is valid. The life you imagined was real to you, and your sadness honors that."),
    ("User", "how do I stop mourning a path I did not take"),
    ("Joe", "Accept that the unchosen path was also uncertain. The life you have now can still hold meaning and beauty."),
]))

blocks.append(make_block([
    ("User", "I feel heavy and weighed down by sadness"),
    ("Joe", "That heaviness is grief seeking expression. Even gentle movement like a short walk can help lighten the load."),
    ("User", "what if the heaviness never lifts"),
    ("Joe", "Sadness does not have to lift all at once. It can gradually shift into a gentler feeling over time."),
]))

blocks.append(make_block([
    ("User", "I am sad because I miss a part of my childhood"),
    ("Joe", "Nostalgia for childhood can bring a bittersweet sadness. Cherishing those memories can also bring warmth."),
    ("User", "what if I cannot find joy in the present"),
    ("Joe", "Joy can return gradually. Be patient with yourself and notice even the smallest moments of brightness."),
]))

blocks.append(make_block([
    ("User", "I feel a hollow emptiness after a loss"),
    ("Joe", "Hollowness after loss is a natural part of grief. Allowing yourself to feel it is more healing than stuffing it down."),
    ("User", "how do I fill that emptiness"),
    ("Joe", "Connection with others and meaningful activities can gradually fill the space, but it takes time and patience."),
]))

# 21-30: Stress and overwhelm
blocks.append(make_block([
    ("User", "I am under so much pressure at work and at home"),
    ("Joe", "When responsibilities stack up, everything feels heavier. Make a short list and tackle one item at a time."),
    ("User", "what if I cannot handle all of my deadlines"),
    ("Joe", "Talk to your manager about prioritizing tasks. Most people respect honesty about capacity more than silent struggle."),
]))

blocks.append(make_block([
    ("User", "I feel trapped in a routine that drains me"),
    ("Joe", "Routines can comfort us, but they can also limit our sense of vitality. Small changes can break the monotony."),
    ("User", "how do I make changes when I feel stuck in a rut"),
    ("Joe", "Pick one small area to change, like a morning walk or a new hobby. Small shifts can create a sense of freshness."),
]))

blocks.append(make_block([
    ("User", "I am stressed about money and bills"),
    ("Joe", "Financial stress is heavy and common. Creating a simple budget can make the situation feel more manageable."),
    ("User", "what if there is no clear way out of debt"),
    ("Joe", "Break the debt into small pieces and focus on paying off the smallest balance first. Each payoff builds momentum."),
]))

blocks.append(make_block([
    ("User", "I feel like I am failing at everything"),
    ("Joe", "Feeling like a failure is not the same as being one. You are trying, and that effort counts for something."),
    ("User", "how do I stop comparing myself to others"),
    ("Joe", "Everyone has their own timeline and challenges. Focus on your own progress rather than measuring against someone else."),
]))

blocks.append(make_block([
    ("User", "I have too many responsibilities and no free time"),
    ("Joe", "Overscheduling leads to burnout. Even ten minutes of downtime can help you recharge and think more clearly."),
    ("User", "how can I protect my time when people keep asking for more"),
    ("Joe", "It is okay to say no or to set boundaries. Protecting your time is not selfish, it is necessary for your wellbeing."),
]))

blocks.append(make_block([
    ("User", "I feel like I am on the edge and about to break"),
    ("Joe", "That feeling of being at the edge means you need rest. Taking even a short break can prevent a full breakdown."),
    ("User", "what if I cannot stop pushing myself"),
    ("Joe", "You have permission to slow down. Asking for help is not a sign of weakness but of wisdom."),
]))

blocks.append(make_block([
    ("User", "I am overwhelmed by the noise and chaos around me"),
    ("Joe", "External chaos can feel internal. Finding quiet moments, even just five minutes, can restore your sense of balance."),
    ("User", "what if I live in a place that is always loud"),
    ("Joe", "Noise-canceling headphones or a quiet corner can create a peaceful space even in a chaotic environment."),
]))

blocks.append(make_block([
    ("User", "I feel like I am carrying the world on my shoulders"),
    ("Joe", "That burden is real but you do not have to carry it alone. Delegating or letting go of some responsibilities can help."),
    ("User", "how do I let go of things I cannot control"),
    ("Joe", "Focus on what you can influence and accept the rest. Control is an illusion that adds unnecessary weight."),
]))

blocks.append(make_block([
    ("User", "I am stressed because my life is not going as planned"),
    ("Joe", "When life deviates from the plan, it can feel destabilizing. Flexibility and self-compassion can help you navigate uncertainty."),
    ("User", "what if my plan never works out"),
    ("Joe", "Life is full of unexpected turns. Some of the best outcomes come from paths you never imagined."),
]))

blocks.append(make_block([
    ("User", "I feel a constant pressure to perform and succeed"),
    ("Joe", "That pressure can come from within or from others. Remember that your worth is not measured by your productivity."),
    ("User", "what if I just cannot meet the expectations"),
    ("Joe", "Not meeting expectations is not a moral failure. It is information about what needs to change."),
]))

# 31-40: Coping strategies
blocks.append(make_block([
    ("User", "I do not know how to cope with intense emotions"),
    ("Joe", "Finding coping strategies takes trial and error. Deep breathing and journaling are good starting points."),
    ("User", "what if nothing seems to help me calm down"),
    ("Joe", "It may take time to find the right tool. Try different approaches and notice which ones give you even a small relief."),
]))

blocks.append(make_block([
    ("User", "I turn to food when I am stressed"),
    ("Joe", "Emotional eating is a common coping mechanism. Recognizing the pattern is the first step toward healthier alternatives."),
    ("User", "what can I do instead of eating when I feel stressed"),
    ("Joe", "Try going for a walk, squeezing a stress ball, or calling a friend. Physical movement can shift your emotional state."),
]))

blocks.append(make_block([
    ("User", "I drink alcohol to deal with my anxiety"),
    ("Joe", "Using substances to cope can create more problems over time. Talking to someone or trying relaxation techniques can be healthier alternatives."),
    ("User", "how do I stop relying on alcohol to feel better"),
    ("Joe", "Start by telling someone you trust about your pattern and explore one new coping skill each week."),
]))

blocks.append(make_block([
    ("User", "I isolate myself when I am feeling down"),
    ("Joe", "Isolation can deepen sadness. Even a brief conversation with a friend can lift your mood."),
    ("User", "what if reaching out feels impossible when I am low"),
    ("Joe", "Start with a simple text message instead of a full conversation. Small steps still count as reaching out."),
]))

blocks.append(make_block([
    ("User", "I have trouble sleeping because my mind will not stop"),
    ("Joe", "A racing mind at night is exhausting. Try writing down your thoughts before bed to get them out of your head."),
    ("User", "what if the worries come back as soon as my head hits the pillow"),
    ("Joe", "Keep a notepad by your bed to capture the worries and tell yourself you can address them tomorrow."),
]))

blocks.append(make_block([
    ("User", "I scratch or pick at my skin when I am stressed"),
    ("Joe", "That is a common stress response called dermatillomania. Awareness and replacement behaviors can help reduce it."),
    ("User", "what if I cannot stop the habit on my own"),
    ("Joe", "Stress reduction techniques and keeping your hands busy can help break the cycle."),
]))

blocks.append(make_block([
    ("User", "I snap at people when I am feeling overwhelmed"),
    ("Joe", "Taking out frustration on others is common but can damage relationships. Acknowledging your triggers is the first step."),
    ("User", "how do I stop being reactive"),
    ("Joe", "Pause before responding and take a breath. That small gap between stimulus and response is where your power lives."),
]))

blocks.append(make_block([
    ("User", "I use scrolling on my phone to numb my feelings"),
    ("Joe", "Numbing with screens can become a cycle. Setting limits and replacing scroll time with a healthier activity can help."),
    ("User", "what if I still reach for my phone when I need comfort"),
    ("Joe", "Try keeping a different comforting object nearby, like a book or a warm drink, to break the automatic pattern."),
]))

blocks.append(make_block([
    ("User", "I yell when I am angry and then feel guilty after"),
    ("Joe", "Feeling guilty after an outburst is common. Understanding your triggers can help you respond differently next time."),
    ("User", "what if I cannot control my volume"),
    ("Joe", "Practice lowering your voice before you reach a yelling level. A softer voice naturally calms the nervous system."),
]))

blocks.append(make_block([
    ("User", "I feel like I need something to calm me down but nothing works"),
    ("Joe", "Finding the right calming method takes experimentation. Try combining physical movement with mindful breathing."),
    ("User", "what if I am running out of ideas"),
    ("Joe", "Consider speaking to a therapist who can introduce new techniques tailored to your needs."),
]))

# 41-50: Empathy and relationships
blocks.append(make_block([
    ("User", "I find it hard to be empathetic toward my partner"),
    ("Joe", "Empathy can be difficult when you are stressed or hurt. Try imagining their perspective before reacting."),
    ("User", "what if I just cannot relate to their feelings"),
    ("Joe", "You do not have to fully understand someone to care about them. Listening without judgment can bridge the gap."),
]))

blocks.append(make_block([
    ("User", "I feel emotionally drained after helping friends all the time"),
    ("Joe", "Being supportive does not mean giving all of yourself. It is okay to set limits so you can sustain your generosity."),
    ("User", "how do I say no without feeling guilty"),
    ("Joe", "You can express care while also honoring your own limits. A simple honest response is always better than silent resentment."),
]))

blocks.append(make_block([
    ("User", "My friend keeps sharing their pain and it exhausts me"),
    ("Joe", "Supporting someone through hardship is demanding. You can be present while also protecting your own emotional energy."),
    ("User", "what if they stop talking to me if I set a boundary"),
    ("Joe", "True friendships can survive honest boundaries. If a friendship ends because you need space, it may have been one-sided."),
]))

blocks.append(make_block([
    ("User", "I want to be more understanding of my family"),
    ("Joe", "Understanding takes patience and often requires seeing things from their perspective. Start by asking what their experience is like."),
    ("User", "what if their experiences are so different from mine"),
    ("Joe", "Different experiences do not mean one is more valid than the other. Curiosity can help you bridge even wide gaps."),
]))

blocks.append(make_block([
    ("User", "I feel like no one in my life truly listens"),
    ("Joe", "Feeling unheard is painful. Try expressing what you need directly, such as saying that you just want to be heard."),
    ("User", "how do I find people who will actually listen"),
    ("Joe", "Look for people who ask you questions and remember details about your life. Quality connections matter more than quantity."),
]))

blocks.append(make_block([
    ("User", "I feel like I am the only one giving in my relationships"),
    ("Joe", "One-sided relationships are draining and unsustainable. It is important to express what you need and observe the response."),
    ("User", "what if the other person does not change"),
    ("Joe", "You cannot force someone to reciprocate, but you can choose to invest in relationships that feel balanced."),
]))

blocks.append(make_block([
    ("User", "I am afraid of being rejected when I show vulnerability"),
    ("Joe", "Vulnerability is courageous but frightening. Sharing small truths first can build trust before deeper disclosure."),
    ("User", "what if someone uses my vulnerability against me"),
    ("Joe", "Healthy people respect your openness. If someone exploits it, that tells you about their character, not your worth."),
]))

blocks.append(make_block([
    ("User", "I struggle to express my needs clearly"),
    ("Joe", "Unclear communication can lead to misunderstandings. Using simple statements like I feel when can help."),
    ("User", "what if I feel embarrassed about stating my needs"),
    ("Joe", "Stating your needs is a skill that improves with practice. Every attempt gets easier than the last."),
]))

blocks.append(make_block([
    ("User", "I keep rescuing people who do not want to help themselves"),
    ("Joe", "Rescuing can enable unhealthy patterns. Supporting someone means being there, not solving their problems for them."),
    ("User", "how do I know the difference between helping and rescuing"),
    ("Joe", "Helping empowers the other person, while rescuing takes away their agency. Ask if they want solutions or just to be heard."),
]))

blocks.append(make_block([
    ("User", "I feel disconnected from my partner even though we are together"),
    ("Joe", "Disconnection in a relationship can stem from unmet emotional needs. Honest conversation can rekindle closeness."),
    ("User", "what if I am too ashamed to bring it up"),
    ("Joe", "Naming the disconnection is a brave first step. It opens the door to a deeper, more authentic connection."),
]))

# 51-60: Self-esteem and self-worth
blocks.append(make_block([
    ("User", "I do not believe I am worthy of love"),
    ("Joe", "That belief is not true. Your worth does not depend on what you achieve or what others think of you."),
    ("User", "how can I start believing that I am enough"),
    ("Joe", "Start by listing things you appreciate about yourself, even small ones. Self-worth grows through consistent self-kindness."),
]))

blocks.append(make_block([
    ("User", "I compare myself to everyone and always come up short"),
    ("Joe", "Comparison steals joy and distorts self-image. Focus on your own journey and celebrate your unique strengths."),
    ("User", "what if it feels impossible to stop comparing"),
    ("Joe", "When you catch yourself comparing, redirect your attention to something you have accomplished recently."),
]))

blocks.append(make_block([
    ("User", "I grew up with criticism and I still hear it in my head"),
    ("Joe", "Internalized criticism from childhood is hard to overcome. You can challenge those inner voices with kinder alternatives."),
    ("User", "how do I silence that critical voice inside my head"),
    ("Joe", "Notice when the voice speaks and ask if that comment would you say to a friend. Replace it with something more compassionate."),
]))

blocks.append(make_block([
    ("User", "I feel confident sometimes and then suddenly doubt myself"),
    ("Joe", "Self-doubt is a normal part of life. Acknowledging both feelings shows emotional maturity and self-awareness."),
    ("User", "what if my self-doubt keeps me from trying new things"),
    ("Joe", "Action can reduce doubt more effectively than thought. Start with a small step and let the confidence follow."),
]))

blocks.append(make_block([
    ("User", "I am working on accepting myself as I am"),
    ("Joe", "Self-acceptance is a journey that involves acknowledging flaws without letting them define you. You are already worthy."),
    ("User", "how do I keep going when self-acceptance feels so far away"),
    ("Joe", "Remember that acceptance does not mean perfection. Each day you choose kindness toward yourself, you move closer."),
]))

blocks.append(make_block([
    ("User", "I feel like I am not good enough for anything"),
    ("Joe", "That voice is not telling the truth. You bring unique value to every situation you participate in."),
    ("User", "what if I just cannot shake this feeling of inadequacy"),
    ("Joe", "Therapy and self-reflection can help you identify where these beliefs came from and how to challenge them."),
]))

blocks.append(make_block([
    ("User", "I feel invisible in my own life"),
    ("Joe", "Feeling invisible is painful and often comes from neglecting your own needs. Making room for yourself is an act of love."),
    ("User", "how do I start noticing myself again"),
    ("Joe", "Check in with yourself daily and ask what you need. Small acts of self-attention can rebuild your sense of presence."),
]))

blocks.append(make_block([
    ("User", "I feel like everyone else has their life together except me"),
    ("Joe", "That impression is misleading. Many people are struggling behind the scenes. Your journey is valid and your own."),
    ("User", "what if I keep falling behind"),
    ("Joe", "Success is not a race. Focus on your own progress and celebrate the steps forward, however small."),
]))

blocks.append(make_block([
    ("User", "I feel like my accomplishments do not count"),
    ("Joe", "Dismissing your achievements is a symptom of low self-worth. Acknowledging them, even with a journal, can shift your perspective."),
    ("User", "what if I cannot see any accomplishments at all"),
    ("Joe", "Start small and look at the day-to-day efforts that keep your life moving. Those are real accomplishments."),
]))

blocks.append(make_block([
    ("User", "I feel like I have to earn the right to exist"),
    ("Joe", "You do not have to earn your right to exist. Being alive is enough. Resting and being present is also valid."),
    ("User", "what if I still feel like a burden to others"),
    ("Joe", "That feeling is a distortion, not a fact. People who care about you want you here, burdens and all."),
]))

# 61-70: Anger and frustration
blocks.append(make_block([
    ("User", "I feel angry at my coworker but I do not know how to say it"),
    ("Joe", "Unexpressed anger builds up and can damage relationships. Calmly stating what bothers you is a healthier approach."),
    ("User", "what if they get defensive when I bring it up"),
    ("Joe", "Stay focused on how their actions made you feel rather than attacking them as a person."),
]))

blocks.append(make_block([
    ("User", "I get frustrated when people do not meet my expectations"),
    ("Joe", "Expectations can lead to disappointment. Communicating your needs clearly can reduce frustration."),
    ("User", "how do I stop expecting so much from others"),
    ("Joe", "Adjust your expectations and express them openly. People cannot read your mind, and that is not their failure."),
]))

blocks.append(make_block([
    ("User", "I have trouble controlling my anger"),
    ("Joe", "Anger management starts with recognizing triggers. Taking a pause before reacting can prevent harmful outbursts."),
    ("User", "what if I lose control even after taking a pause"),
    ("Joe", "If you still lose control, apologize and reflect on what triggered it. Each attempt is progress toward better management."),
]))

blocks.append(make_block([
    ("User", "I feel angry at myself for making the same mistake"),
    ("Joe", "Self-directed anger does not help you grow. Accept the mistake, learn from it, and move forward."),
    ("User", "how do I forgive myself for repeating the same error"),
    ("Joe", "Forgiveness is a process. Treat yourself with the same patience you would give a friend in the same situation."),
]))

blocks.append(make_block([
    ("User", "I am furious about something unfair that happened to me"),
    ("Joe", "Feeling furious about unfairness is natural. Channel that energy into constructive action or honest expression."),
    ("User", "what if nothing I do changes the unfair situation"),
    ("Joe", "You can still control your response and how you treat yourself. Your value is not determined by external fairness."),
]))

blocks.append(make_block([
    ("User", "I feel angry that my feelings are being dismissed"),
    ("Joe", "Having your feelings dismissed is hurtful and invalidating. Your emotions are real and deserve to be acknowledged."),
    ("User", "how do I stand up for my feelings without being aggressive"),
    ("Joe", "State clearly what you feel and why it matters to you. Assertiveness does not require raising your voice."),
]))

blocks.append(make_block([
    ("User", "I feel a hot rage when someone breaks a promise"),
    ("Joe", "Broken promises can trigger deep anger. Expressing that hurt clearly can help you address the betrayal."),
    ("User", "what if they just keep breaking promises"),
    ("Joe", "Repeated broken promises signal a lack of respect. You have every right to distance yourself from such people."),
]))

blocks.append(make_block([
    ("User", "I feel angry at the world for not being fair"),
    ("Joe", "Anger at injustice is a valid response. Channeling that anger into advocacy or creative expression can be powerful."),
    ("User", "what if the anger never goes away"),
    ("Joe", "Anger can be a signal that something needs changing. Use it as fuel rather than letting it consume you."),
]))

blocks.append(make_block([
    ("User", "I feel like I am angry all the time and I cannot figure out why"),
    ("Joe", "Chronic anger can mask deeper feelings like hurt or fear. Exploring those underlying emotions can bring clarity."),
    ("User", "what if the underlying feelings are too painful to face"),
    ("Joe", "A therapist can help you gently explore those deeper feelings at a pace that feels safe."),
]))

blocks.append(make_block([
    ("User", "I am angry that I cannot be happy"),
    ("Joe", "Being angry at your inability to feel happy adds another layer of pain. Accepting your current state is a step toward healing."),
    ("User", "how do I stop beating myself up for not being happy"),
    ("Joe", "Happiness is not a requirement for a good life. Allow yourself to feel whatever comes without judgment."),
]))

# 71-80: Loneliness and connection
blocks.append(make_block([
    ("User", "I moved to a new city and feel completely alone"),
    ("Joe", "Starting over in a new place is isolating. Joining a group or class can help you meet people with shared interests."),
    ("User", "what if I try to make friends and they all reject me"),
    ("Joe", "Rejection is painful but it does not define your worth. Keep showing up authentically and the right people will come."),
]))

blocks.append(make_block([
    ("User", "I am lonely even though I am in a relationship"),
    ("Joe", "Loneliness within a relationship is valid. It may signal a need for deeper emotional connection."),
    ("User", "how do I talk to my partner about feeling lonely"),
    ("Joe", "Use honest statements about your feelings rather than blame, such as saying you miss feeling close."),
]))

blocks.append(make_block([
    ("User", "I have friends but still feel empty inside"),
    ("Joe", "Having company does not always mean having connection. Try sharing something more personal to deepen the bonds."),
    ("User", "what if I am afraid to be vulnerable with them"),
    ("Joe", "Vulnerability is frightening but it is the foundation of true connection. Start with one small honest disclosure."),
]))

blocks.append(make_block([
    ("User", "I feel disconnected from my own emotions"),
    ("Joe", "Emotional disconnection can come from suppressing feelings. Naming your emotions can help you reconnect with yourself."),
    ("User", "how do I figure out what I am actually feeling"),
    ("Joe", "Try checking in with your body and asking what sensation you notice. Feelings often show up physically before they are clear."),
]))

blocks.append(make_block([
    ("User", "I miss the close friendships I had in college"),
    ("Joe", "Nostalgia for old connections is natural, but current relationships can also be deeply meaningful."),
    ("User", "what if I cannot make new friends as close as those were"),
    ("Joe", "Every friendship has its own timeline and depth. Focus on building what you have now rather than comparing."),
]))

blocks.append(make_block([
    ("User", "I feel like I have no one to call at 3 AM"),
    ("Joe", "That feeling of having no one to reach out to is deeply lonely. Building deeper connections takes time but is possible."),
    ("User", "what if I never find that kind of closeness"),
    ("Joe", "Close friendships develop gradually through consistent honesty and shared experiences. Keep being open to new connections."),
]))

blocks.append(make_block([
    ("User", "I feel surrounded by people but still completely alone"),
    ("Joe", "Being alone in a crowd is a painful form of loneliness. Seeking meaningful interaction rather than surface-level contact can help."),
    ("User", "what if small talk feels like too much effort"),
    ("Joe", "You do not have to make small talk. Even one honest question can open a door to a more genuine connection."),
]))

blocks.append(make_block([
    ("User", "I feel disconnected from my family growing up"),
    ("Joe", "Family disconnection can create lasting loneliness. Setting boundaries or seeking therapy can help heal those wounds."),
    ("User", "what if my family does not want to reconnect"),
    ("Joe", "You cannot control their willingness, but you can take steps to heal your own sense of belonging and self-worth."),
]))

blocks.append(make_block([
    ("User", "I feel like an outsider everywhere I go"),
    ("Joe", "Feeling like an outsider can be exhausting. Finding your community or creating your own space can help you feel at home."),
    ("User", "what if I never truly fit in anywhere"),
    ("Joe", "You do not have to fit in anywhere. Belonging to yourself is the most important form of belonging."),
]))

blocks.append(make_block([
    ("User", "I am lonely because I have been through things no one else understands"),
    ("Joe", "Carrying experiences that few can relate to can feel isolating. Support groups or therapists can offer understanding."),
    ("User", "what if I cannot find anyone who understands"),
    ("Joe", "Even if no one has had the exact experience, many people understand deep pain and can offer genuine empathy."),
]))

# 81-90: Hope and recovery
blocks.append(make_block([
    ("User", "I am trying to stay hopeful despite a difficult diagnosis"),
    ("Joe", "Hope is a powerful force even in hard times. Focusing on small daily joys can sustain your spirit."),
    ("User", "what if the hope feels fake and I do not believe it"),
    ("Joe", "Hope does not have to be strong or confident to be real. Even a tiny spark of hope counts."),
]))

blocks.append(make_block([
    ("User", "I am recovering from addiction and I am scared of relapse"),
    ("Joe", "Recovery is not linear and fear of relapse is normal. Having a plan and support system can help you stay on track."),
    ("User", "what if I slip and fall back into old patterns"),
    ("Joe", "A slip is not a failure. It is a learning opportunity. Reach out and get back on track as soon as possible."),
]))

blocks.append(make_block([
    ("User", "I feel like I am making progress but it is so slow"),
    ("Joe", "Progress in mental health is often gradual and not always visible. Trust the process and notice even tiny steps."),
    ("User", "how do I stay patient when healing takes so long"),
    ("Joe", "Healing has its own timeline. Comparing your progress to others only adds unnecessary pressure."),
]))

blocks.append(make_block([
    ("User", "I want to forgive someone who hurt me deeply"),
    ("Joe", "Forgiveness is for your own peace, not a gift to the person who hurt you. It is a process, not an instant decision."),
    ("User", "what if forgiving feels like letting them off the hook"),
    ("Joe", "Forgiving does not mean excusing the behavior or reconciling. It means releasing the weight of resentment."),
]))

blocks.append(make_block([
    ("User", "I am rebuilding my life after a painful breakup"),
    ("Joe", "Rebuilding takes time and self-compassion. Focus on rediscovering who you are outside of that relationship."),
    ("User", "what if I never feel happy again"),
    ("Joe", "Happiness can return, but it may look different than before. Give yourself permission to define it on your own terms."),
]))

blocks.append(make_block([
    ("User", "I am learning to trust myself again after being betrayed"),
    ("Joe", "Trusting yourself again is a gradual process. Listening to your own instincts is the first step toward rebuilding trust."),
    ("User", "what if I never trust anyone again"),
    ("Joe", "Trusting yourself does not mean trusting everyone else. You can be cautious while still remaining open to genuine people."),
]))

blocks.append(make_block([
    ("User", "I feel like I have lost my sense of purpose"),
    ("Joe", "Losing purpose can happen during transitions. Exploring what once brought you joy can help you find direction again."),
    ("User", "what if I cannot find a new purpose"),
    ("Joe", "Purpose can emerge slowly from small daily actions and connections. It does not have to be grand to be meaningful."),
]))

blocks.append(make_block([
    ("User", "I am hopeful about the future but also scared"),
    ("Joe", "Having mixed feelings about the future is normal. Both hope and fear can coexist without canceling each other out."),
    ("User", "what if my fear wins out over my hope"),
    ("Joe", "Fear does not have to win. You can choose to act on hope even while acknowledging the fear."),
]))

blocks.append(make_block([
    ("User", "I am trying to recover from a toxic relationship"),
    ("Joe", "Recovering from toxicity takes time and self-compassion. You are learning to trust yourself again."),
    ("User", "what if the trauma from that relationship follows me"),
    ("Joe", "Trauma responses are normal and treatable. Therapy can help you process the experience and move forward."),
]))

blocks.append(make_block([
    ("User", "I feel like I am coming back to life after a long winter"),
    ("Joe", "Emerging from a difficult period feels like rebirth. Celebrate the small victories that mark your return."),
    ("User", "what if I am afraid of the light after being in the dark"),
    ("Joe", "It is normal to feel hesitant after prolonged darkness. Take it slowly and let your eyes adjust to the brightness."),
]))

# 91-100: More anxiety and worry topics
blocks.append(make_block([
    ("User", "I have panic attacks that come out of nowhere"),
    ("Joe", "Panic attacks feel terrifying but they are not dangerous. Focusing on slow breathing can help them pass."),
    ("User", "what if I have one in a public place"),
    ("Joe", "Panic attacks end, even in public. Find a quiet spot and remember that this feeling is temporary."),
]))

blocks.append(make_block([
    ("User", "I worry about my health constantly"),
    ("Joe", "Health anxiety can be paralyzing. Schedule regular checkups to get reassurance from professionals rather than searching online."),
    ("User", "what if I find something wrong that I cannot afford to treat"),
    ("Joe", "Most health worries turn out to be nothing serious. Focus on what you can control, like lifestyle and early detection."),
]))

blocks.append(make_block([
    ("User", "I am anxious about my child's safety"),
    ("Joe", "Worrying about your child is a parent's natural instinct. Balancing vigilance with trust is important for both of you."),
    ("User", "how do I stop scaring myself with worst-case scenarios"),
    ("Joe", "Remind yourself that most outcomes are safe and that your child is resilient. Focus on what you can do today to support them."),
]))

blocks.append(make_block([
    ("User", "I feel anxious when I have to make decisions"),
    ("Joe", "Decision anxiety is common. Breaking choices into smaller steps can make them feel less overwhelming."),
    ("User", "what if I make the wrong decision and regret it"),
    ("Joe", "Most decisions can be adjusted. Even a wrong choice teaches you something useful for the next one."),
]))

blocks.append(make_block([
    ("User", "My anxiety makes me avoid situations I enjoy"),
    ("Joe", "Avoidance reinforces anxiety. Taking small steps toward what you fear can help desensitize you over time."),
    ("User", "how do I face my fears without being overwhelmed"),
    ("Joe", "Start with the least scary version of the situation and gradually increase the challenge as you build confidence."),
]))

blocks.append(make_block([
    ("User", "I feel anxious about my parents aging and getting sick"),
    ("Joe", "Worrying about aging parents is natural and shows how much you care. Focusing on quality time now can ease the fear."),
    ("User", "what if there is nothing I can do to help them stay healthy"),
    ("Joe", "You can provide love, support, and presence. Medical care is in their hands, but your kindness matters greatly."),
]))

blocks.append(make_block([
    ("User", "I feel anxious about my financial future"),
    ("Joe", "Financial anxiety is common and understandable. Creating a plan and setting small savings goals can reduce the uncertainty."),
    ("User", "what if the economy gets worse and I lose everything"),
    ("Joe", "Economic downturns are cyclical and temporary. Building resilience through savings and skills can weather any storm."),
]))

blocks.append(make_block([
    ("User", "I am anxious about starting a new chapter in my life"),
    ("Joe", "Starting fresh is exciting and scary at the same time. Embracing the uncertainty can make the transition smoother."),
    ("User", "what if I make a mistake in this new chapter"),
    ("Joe", "Mistakes are part of every new beginning. They teach you what works and what does not."),
]))

blocks.append(make_block([
    ("User", "I feel like my anxiety is getting worse instead of better"),
    ("Joe", "Anxiety can fluctuate and sometimes worsen before it improves. This does not mean treatment is not working."),
    ("User", "what if I am just stuck with this forever"),
    ("Joe", "Anxiety is treatable and manageable. Adjusting your approach with professional guidance can make a significant difference."),
]))

blocks.append(make_block([
    ("User", "I feel anxious when people are counting on me"),
    ("Joe", "The weight of others expectations can be overwhelming. Communicating your limits can reduce that pressure."),
    ("User", "what if they expect me to deliver no matter what"),
    ("Joe", "You are not obligated to meet every expectation. It is okay to say what you can realistically handle."),
]))

# 101-110: More sadness and emotional topics
blocks.append(make_block([
    ("User", "I feel melancholy on days with no sunshine"),
    ("Joe", "Seasonal mood shifts are real. Getting light exposure or using a lamp can help stabilize your mood."),
    ("User", "what if the sadness persists even when the sun comes out"),
    ("Joe", "Some sadness is not tied to weather. If it persists, talking to a professional can help identify the underlying cause."),
]))

blocks.append(make_block([
    ("User", "I feel like my emotions are on a roller coaster"),
    ("Joe", "Emotional volatility can be exhausting. Identifying triggers and building steady routines can smooth the ride."),
    ("User", "what if I cannot figure out what is causing the swings"),
    ("Joe", "Keep a brief daily journal to track patterns. Sometimes the causes become visible only when you look back."),
]))

blocks.append(make_block([
    ("User", "I am sad because I will never have children"),
    ("Joe", "That is a real loss and your grief is valid. Finding alternative ways to nurture and create meaning can bring comfort."),
    ("User", "how do I stop feeling like my life is incomplete"),
    ("Joe", "Your life has value regardless of whether it follows a traditional path. Explore what brings you fulfillment now."),
]))

blocks.append(make_block([
    ("User", "I feel a deep sense of emptiness inside"),
    ("Joe", "Emptiness can be a signal that something is missing or unprocessed. Exploring that feeling with curiosity can be a start."),
    ("User", "what if I feel empty even when I am doing things I enjoy"),
    ("Joe", "That is a sign that the activities may not connect to your deeper needs. Try exploring what truly matters to you."),
]))

blocks.append(make_block([
    ("User", "I am mourning a relationship that ended years ago"),
    ("Joe", "Grief has no expiration date. It is okay to still feel sadness even after a long time."),
    ("User", "what if I should have moved on by now"),
    ("Joe", "There is no timeline for grief. Allow yourself to feel whatever comes without judging yourself for it."),
]))

blocks.append(make_block([
    ("User", "I feel sad when I see others being happy"),
    ("Joe", "Seeing others happy while you struggle can deepen sadness. Remember that everyone has hidden struggles of their own."),
    ("User", "what if I cannot stop feeling envious"),
    ("Joe", "Envy is a signal that you want something for yourself. Redirect that desire into actions that serve your own growth."),
]))

blocks.append(make_block([
    ("User", "I feel a deep sadness about the state of my community"),
    ("Joe", "Sadness about collective struggles shows your empathy and connection. Channeling that into action can help."),
    ("User", "what if my efforts to help feel inadequate"),
    ("Joe", "Every act of care matters, even if it feels small. Compassion ripples outward in ways you cannot always see."),
]))

blocks.append(make_block([
    ("User", "I feel sad about not living up to my potential"),
    ("Joe", "That disappointment shows you have aspirations. Redefining what potential means can ease the self-judgment."),
    ("User", "what if I never reach that potential"),
    ("Joe", "Potential is not a single destination. Growth is a process, and every step forward has its own value."),
]))

blocks.append(make_block([
    ("User", "I feel sorrow when I think about the choices I did not make"),
    ("Joe", "Wondering about unchosen paths is a universal experience. The road not taken may have had its own difficulties."),
    ("User", "what if I made the wrong choice"),
    ("Joe", "Every choice had its reasons at the time. There is no way to know the other path without actually walking it."),
]))

blocks.append(make_block([
    ("User", "I feel a quiet sadness that I cannot explain"),
    ("Joe", "Sometimes sadness has no obvious cause. It is okay to feel sad even when you cannot name the reason."),
    ("User", "what if this quiet sadness never lifts"),
    ("Joe", "Quiet sadness can be a signal to slow down and check in with yourself. Small acts of care can help it shift."),
]))

# 111-120: More coping and resilience
blocks.append(make_block([
    ("User", "I am learning to set boundaries but it feels selfish"),
    ("Joe", "Boundaries are an act of self-care, not selfishness. They actually help you show up better for others."),
    ("User", "what if people get upset that I am saying no"),
    ("Joe", "Their discomfort does not mean you did something wrong. Healthy relationships can adjust to mutual respect."),
]))

blocks.append(make_block([
    ("User", "I use humor to avoid dealing with painful feelings"),
    ("Joe", "Humor can be a healthy coping tool when it does not become an escape. Recognize when it is keeping you from processing."),
    ("User", "what if I do not know when my humor is just avoidance"),
    ("Joe", "Notice if you feel more comfortable making jokes than sharing real feelings. That pattern is a sign to slow down."),
]))

blocks.append(make_block([
    ("User", "I am trying mindfulness meditation but my mind wanders"),
    ("Joe", "A wandering mind is normal in meditation. Gently redirecting your attention is the practice itself."),
    ("User", "what if I never get better at focusing"),
    ("Joe", "Meditation is not about perfect focus. Each time you notice and redirect, you are strengthening your attention."),
]))

blocks.append(make_block([
    ("User", "I feel exhausted from always being strong for everyone"),
    ("Joe", "Carrying everyone else's burdens is unsustainable. It is okay to let others carry some weight too."),
    ("User", "how do I let people know I need support without feeling weak"),
    ("Joe", "Asking for support shows strength and gives others the chance to care about you."),
]))

blocks.append(make_block([
    ("User", "I am practicing self-compassion and it feels awkward"),
    ("Joe", "Self-compassion is a new skill for many people. The awkwardness is just part of learning something unfamiliar."),
    ("User", "what if I feel like I am faking it"),
    ("Joe", "It does not have to feel genuine to be beneficial. Over time, self-kindness becomes more natural."),
]))

blocks.append(make_block([
    ("User", "I journal every day but I feel like it is not helping"),
    ("Joe", "Journaling can be powerful but it is not always immediately effective. Try different prompts or approaches."),
    ("User", "what if I keep writing and nothing changes"),
    ("Joe", "Writing is a process. The insights may come gradually. Trust the practice and keep going."),
]))

blocks.append(make_block([
    ("User", "I exercise to cope but I am afraid I am overdoing it"),
    ("Joe", "Exercise is healthy but compulsive exercise can be a coping mechanism that masks deeper issues."),
    ("User", "how do I know if my exercise is healthy coping or avoidance"),
    ("Joe", "Check if you are exercising to escape feelings or to care for your body. Both can be valid, but balance matters."),
]))

blocks.append(make_block([
    ("User", "I am trying to replace negative self-talk with positive affirmations"),
    ("Joe", "Affirmations can be helpful but they must feel believable to be effective. Start with realistic statements."),
    ("User", "what if the affirmations feel fake and I do not believe them"),
    ("Joe", "Use evidence-based statements rather than sweeping claims. Gradually the truth behind them will feel real."),
]))

blocks.append(make_block([
    ("User", "I have a support system but still feel like coping alone"),
    ("Joe", "Even with support, it can feel easier to handle things alone. Accepting help is a strength, not a weakness."),
    ("User", "what if I am burdening the people who care about me"),
    ("Joe", "People who genuinely care want to be there for you. Letting them help is a gift to them as well as to you."),
]))

blocks.append(make_block([
    ("User", "I am trying creative outlets to process my emotions"),
    ("Joe", "Creative expression can be a powerful way to process feelings that are hard to put into words."),
    ("User", "what if I am not talented enough to create anything meaningful"),
    ("Joe", "Creativity is not about talent or perfection. It is about expressing what is inside, regardless of the result."),
]))

# 121-130: Final batch
blocks.append(make_block([
    ("User", "I feel guilty for not spending enough time with my family"),
    ("Joe", "Guilt from competing responsibilities is common. Quality time matters more than quantity."),
    ("User", "how do I make up for the time I missed"),
    ("Joe", "Start now with focused, present moments. Even a short conversation where you are fully there can heal a lot."),
]))

blocks.append(make_block([
    ("User", "I am scared to open up about my trauma"),
    ("Joe", "Sharing trauma takes immense courage. When you are ready, a trusted person or therapist can hold that safely."),
    ("User", "what if no one is ready to hear what I have been through"),
    ("Joe", "You do not have to share everything at once. Start with what feels manageable and build trust over time."),
]))

blocks.append(make_block([
    ("User", "I feel like I am pretending to be okay all the time"),
    ("Joe", "Pretending to be okay is exhausting. Being honest about your real feelings is the first step toward healing."),
    ("User", "how do I admit that I am not okay"),
    ("Joe", "You can start with one trusted person or even write it down. Acknowledging the truth is the beginning of change."),
]))

blocks.append(make_block([
    ("User", "I am grieving a miscarriage and I feel alone"),
    ("Joe", "Losing a wanted pregnancy is a profound loss that is often overlooked. Your grief is valid and deserves space."),
    ("User", "what if others dismiss my pain because it was early"),
    ("Joe", "The timing of a loss does not determine its significance. Your feelings are real and deserve to be honored."),
]))

blocks.append(make_block([
    ("User", "I feel like I am running on empty every single day"),
    ("Joe", "Running on empty leads to burnout. Even small acts of rest and recharge can rebuild your energy reserves."),
    ("User", "what if there is no time in my schedule to rest"),
    ("Joe", "Rest does not always require free time. A few minutes of stillness or a short walk can be restorative."),
]))

blocks.append(make_block([
    ("User", "I am angry that my feelings are being dismissed"),
    ("Joe", "Having your feelings dismissed is hurtful and invalidating. Your emotions are real and deserve to be acknowledged."),
    ("User", "how do I stand up for my feelings without being aggressive"),
    ("Joe", "State clearly what you feel and why it matters to you. Assertiveness does not require raising your voice."),
]))

blocks.append(make_block([
    ("User", "I feel hopeless about the state of the world"),
    ("Joe", "Global concerns can feel overwhelming. Contributing locally or focusing on what you can influence can restore hope."),
    ("User", "what if my efforts feel too small to make a difference"),
    ("Joe", "Small actions create ripples. One person caring can inspire others to care as well."),
]))

blocks.append(make_block([
    ("User", "I am struggling with low self-esteem that has lasted for years"),
    ("Joe", "Long-standing low self-esteem is deeply rooted but can be changed with patience and consistent self-work."),
    ("User", "what if this is just who I am and cannot be changed"),
    ("Joe", "Your self-image is not fixed. With support and effort, your sense of worth can grow over time."),
]))

blocks.append(make_block([
    ("User", "I feel a strange calm about ending a long friendship"),
    ("Joe", "Sometimes ending a relationship feels right even when it is painful. Trust that feeling and honor your needs."),
    ("User", "what if I regret this decision later"),
    ("Joe", "Regret does not mean you made the wrong choice. You are allowed to grow and outgrow relationships."),
]))

blocks.append(make_block([
    ("User", "I am trying to be more present and stop overthinking"),
    ("Joe", "Overthinking pulls you out of the present moment. Noticing your surroundings can ground you right now."),
    ("User", "what if I cannot stop my thoughts from spiraling"),
    ("Joe", "Thoughts are not facts. Simply observing them without getting caught in them can reduce their power."),
]))

# Write the file
os.makedirs("/Users/dev/github-projects/joe-brain/data/_gen/emotion", exist_ok=True)

with open("/Users/dev/github-projects/joe-brain/data/_gen/emotion/emotion_mt_07_cand.txt", "w") as f:
    for i, block in enumerate(blocks):
        f.write(block)
        if i < len(blocks) - 1:
            f.write("\n\n")
        else:
            f.write("\n")

# Verify
with open("/Users/dev/github-projects/joe-brain/data/_gen/emotion/emotion_mt_07_cand.txt") as f:
    content = f.read()

block_list = content.strip().split("\n\n")
print(f"Total blocks written: {len(block_list)}")

multi_turn = 0
for block in block_list:
    turns = block.strip().split("\n")
    user_count = sum(1 for t in turns if t.startswith("User: "))
    if user_count >= 2:
        multi_turn += 1

single_turn = len(block_list) - multi_turn
print(f"Multi-turn blocks: {multi_turn} ({100 * multi_turn / len(block_list):.1f}%)")
print(f"Single-turn blocks: {single_turn} ({100 * single_turn / len(block_list):.1f}%)")

# Verify ASCII only
non_ascii = []
for i, block in enumerate(block_list):
    for ch in block:
        if ord(ch) > 127:
            non_ascii.append((i, ch))
if non_ascii:
    print(f"WARNING: Found non-ASCII characters: {non_ascii[:5]}")
else:
    print("All characters are ASCII")

# Check no smart quotes, em-dash, tilde, or accented chars
forbidden = ["\u201c", "\u201d", "\u2014", "~"]
found_forbidden = []
for i, block in enumerate(block_list):
    for f_char in forbidden:
        if f_char in block:
            found_forbidden.append((i, repr(f_char)))
if found_forbidden:
    print(f"WARNING: Found forbidden characters: {found_forbidden[:5]}")
else:
    print("No smart quotes, em-dash, or tildes found")

# Verify format: each block starts with User:
format_ok = True
for i, block in enumerate(block_list):
    lines = block.strip().split("\n")
    if not lines[0].startswith("User: "):
        format_ok = False
        print(f"Block {i} does not start with User: ")
if format_ok:
    print("All blocks start with User: - format OK")

# Verify each block has blank line separation (already checked by split)
print("Format verification complete")
PYEOF