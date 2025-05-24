 #include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std;

int main() {
    srand(time(0));
    string start;
    cout << "\t\t\t\t\t\tstart the game ";
    cin >> start;
    int numPlayers;
    cout << "\t\t\t\t\tEnter number of players: ";
    cin >> numPlayers;
    cin.ignore();

    vector<string> playerNames(numPlayers);
    vector<int> playerBalance(numPlayers, 100);

    for (int i = 0; i < numPlayers; ++i) {
        cout << "\t\t\t\t\tEnter name for Player " << (i + 1) << ": ";
        cin >> playerNames[i];
    }
    vector <string> truth = {
" What’s the biggest secret you’re hiding ?",
"Have you ever loved someone who didn’t love you back ?",
"What’s the most embarrassing thing from your childhood ?",
    "Did you lie today ?",
"	Who do you trust the most ?",
"Have you ever betrayed a friend ?",
"	Who was the last person you searched for online ?",
"	Are you hiding anything from your family ?",
    "What was your most embarrassing moment ?",
"	Have you ever cried in front of someone you didn’t want to ?",
    "What’s something you deeply regret ?",
"	Who was your first crush ?"	,
"Have you ever thought about running away from everything ?",
"	Who do you hate the most and why ?",
"	What’s something you’re scared others will find out about you ?",
"	Have you ever regretted a friendship ?",
"   What’s the worst decision you’ve ever made ?",
"	What makes you extremely jealous ?",
"	Have you ever secretly wanted someone to fail ?",
"	Do you secretly stalk someone on social media ?",
"	What’s the most humiliating thing that ever happened to you ?",
"Are you in love right now ?",
"	Do you prefer love or money ?",
"	Are you always honest ?",
"	Have you ever faked a smile to hide sadness ?",
"	Do you overthink the past ?",
"	What’s your biggest weakness ?",
"	Who is the one person you’ll never forget ?",
"	Is there a secret you’re still afraid to reveal ?",
 "What is your biggest fear?",
    "Have you ever lied to your best friend?",
    "Do you have a secret crush?",
    "What is something embarrassing you’ve done?",
    "Have you ever failed on purpose?",
    "What’s a lie you’ve told recently?",
    "Have you ever cheated in a game?",
    "Who do you love the most in your life?",
    "What is one thing you’re hiding from your parents?",
    "Have you ever broken someone’s trust?",
    "What is the most childish thing you still do?",
    "Have you ever cried because of a movie?",
    "What is something you’re ashamed of?",
    "Have you ever stalked someone online?",
    "What’s the most dangerous thing you’ve done?",
    "Have you ever pretended to like someone?",
    "What’s your guilty pleasure?",
    "Have you ever been jealous of a friend?",
    "What’s the worst lie you've ever told?",
    "If you had a chance to erase one memory, what would it be?",
"What is one thing you've done that you never told anyone?",
"Have you ever had feelings for someone forbidden(like a teacher or a best friend’s partner) ?",
"If you could delete one person from your life without consequences, who would it be ?",
"What’s a dark thought you’ve had but never said out loud ?",
"Have you ever wished something bad happened to someone close to you ?",
"What is the biggest lie you told in a relationship ?",
"Have you ever betrayed someone who trusted you ?",
"What’s the worst thing you’ve done out of jealousy ?",
"Who is the person you pretend to like but actually don’t ?",
"What’s your deepest fear about the future ?"
    };

    vector <string> dare = {
   "Sing a part of your favorite song out loud.",
"Call a friend and say “I love you” then hang up.",
"Speak in a different accent for 3 minutes.",
"Walk like a runway model around the room.",
"Take a weird selfie and show it to everyone.",
"Eat a spoonful of salt or lemon.",
"Share an embarrassing story about yourself.",
"Act out a dramatic movie scene.",
"	Imitate someone in the room.",
"Talk for one minute without closing your mouth.",
"Post a silly sentence on your WhatsApp or Facebook status.",
"	Draw a face on your hand and make it talk.",
"	Walk on your tiptoes for a full lap in the room.",
"	Show the group your most recent photo in your gallery.",
"	Wear something ridiculous for 5 minutes.",
"Put on makeup (if you’re a guy).",
"	Tell a really bad joke and try to make someone laugh."
"	Let someone feed you with your eyes closed.",
"	Show the first image from your photo gallery.",
"	Pretend to be a panda bear for one minute.",
"	Speak like a robot for two minutes.",
"	Make an animal noise and let others guess it.",
"	Jump 10 times and say “I’m a ninja” loudly.",
"	Send “I have a crush on you” to someone random in your contacts.",
"	Change your name on WhatsApp to something silly for one hour.",
"	Talk using only backward sentences.",
"	Say “I want candy” five times in a baby voice.",
"	Send a random gibberish message to someone in your contacts.",
"	Sing a commercial jingle like it’s an opera song.",
"	Write “I’m the legend” on your hand and keep it visible.",
     "Talk in a funny voice!",
 "Sing a song loudly in front of everyone.",
    "Act like a chicken for 30 seconds.",
    "Speak with your eyes closed for one minute.",
    "Dance without music for 20 seconds.",
    "Try to touch your nose with your tongue.",
    "Send a funny emoji to the last person you texted.",
    "Do 10 pushups right now.",
    "Pretend to cry like a baby.",
    "Call someone and sing 'Happy Birthday'.",
    "Make the weirdest face you can and take a selfie.",
    "Talk like a robot for the next turn.",
    "Do your best celebrity impression.",
    "Walk backwards across the room.",
    "Draw a heart on your cheek with your finger.",
    "Say the alphabet backward.",
    "Wrap yourself in a blanket like a burrito.",
    "Let another player post something on your status.",
    "Say everything twice until your next turn.",
    "Pretend to be a cat and meow 5 times.",
    "Give a compliment to each player in the game.",
"     Send “I love you 😍” to a random contact and don’t explain.",
"Let the group look through your photo gallery for 30 seconds.",
"Post “I miss you ❤️” on your story without any name.",
"Eat a mix of 3 weird things chosen by others.",
"Do a fake phone call confessing love to someone.",
"Give your phone to another player to send any message they want.",
"Imitate a celebrity and act for 1 minute.",
"Wear your clothes inside out for the rest of the game.",
"Let someone draw something on your face with a pen.",
"Reveal the last 5 people you chatted with.",

    };


    int choice;
    string answer;
    int currentPlayer = 0;

    while (true) {
        cout << "\n🎮 It's " << playerNames[currentPlayer] << "'s turn!";
        cout << "\n💰 Balance: " << playerBalance[currentPlayer] << " Rupees\n";

        cout << "\n1. Truth\n2. Dare\n3. Exit Game\nChoose (1-3): ";
        cin >> choice;
        cin.ignore();

        if (choice == 1) {
            int i = rand() % truth.size();
            cout << "\nTruth: " << truth[i] << "\n";
            cout << "Will you answer? (yes/no): ";
            cin >> answer;
            if (answer == "yes" || answer == "Yes") {
                playerBalance[currentPlayer] += 20;
                cout << "✔ You earned 20 Rupees.\n";
            }
            else {
                playerBalance[currentPlayer] -= 20;
                cout << "✘ 20 Rupees deducted.\n";
            }
        }
        else if (choice == 2) {
            int i = rand() % dare.size();
            cout << "\nDare: " << dare[i] << "\n";
            cout << "Will you do it? (yes/no): ";
            cin >> answer;
            if (answer == "yes" || answer == "Yes") {
                playerBalance[currentPlayer] += 20;
                cout << "✔ You earned 20 Rupees.\n";
            }
            else {
                playerBalance[currentPlayer] -= 20;
                cout << "✘ 20 Rupees deducted.\n";
            }
        }
        else if (choice == 3) {
            cout << "\n🎮 Game Over! Final Scores:\n";
            for (int i = 0; i < numPlayers; ++i) {
                cout << playerNames[i] << ": " << playerBalance[i] << " Rupees\n";
            }
            break;
        }
        else {
            cout << "Invalid choice. Try again.\n";
        }

        currentPlayer = (currentPlayer + 1) % numPlayers;
    }

    return 0;
}



