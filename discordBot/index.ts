const { Client, Events, GatewayIntentBits } = require("discord.js");
const { OpenAI } = require("openai");

const DISCORD_TOKEN = process.env.DISCORD_TOKEN;
const username = "" || process.env.USERNAME;
const SYSTEM_PROMPT = `Your name is ${username}, reply freely to the provided messages in their style. Messages are formatted as 'Author: Content'.`;
const MODEL_API_URL = "" || process.env.MODEL_API_URL;
const MODEL_NAME = "" || process.env.MODEL_NAME;

// intends + login
const discordClient = new Client({
	intents: [
		GatewayIntentBits.Guilds,
		GatewayIntentBits.GuildMessages,
		GatewayIntentBits.MessageContent,
	],
});

discordClient.on(Events.ClientReady, () => {
	console.log(`Logged in as ${discordClient.user.tag}`);
});
discordClient.login(DISCORD_TOKEN);

let openai: any;

const replaceEmojis = (msg: string) => {
	const emojiRegex = /:(\w+):/g;
	return msg.replace(emojiRegex, (match, emojiName) => {
		const emoji = discordClient.emojis.cache.find(
			(e: any) => e.name === emojiName,
		);
		if (emoji) {
			return `<:${emoji.name}:${emoji.id}>`;
		}
		return match;
	});
};

const getOpenAIInstance = () => {
	if (!openai) {
		openai = new OpenAI({
			baseURL: MODEL_API_URL,
		});
	}
	return openai;
};

const generateResponse = async (messages: any[]) => {
	const openai = getOpenAIInstance();
	const completion = await openai.chat.completions.create({
		model: MODEL_NAME,
		messages: messages,
        repetition_penalty: 1.05
	});
	return completion.choices[0].message.content;
};

const formatMessages = (messages: any[]) => {
    const system = { role: "system", content: SYSTEM_PROMPT + " /no_think" };
    const formattedMessages = messages.map((msg) => {
        if (msg.author.username === discordClient.user.username) {
            return { role: "assistant", content: msg.content.trim() };
        } else {
            return { role: "user", content: msg.author.username + ': ' + msg.content.trim() };
        }
    });

    return [system, ...(formattedMessages.reverse())];
};

const fetchMessageData = async (channel: any) => {
	const options = { limit: 3 };
	return await channel.messages.fetch(options);
};

discordClient.on(Events.MessageCreate, async (msg: any) => {
	try {
		const tagPrefix = "<@" + discordClient.user.id + ">";
		if (!("guild" in msg) || !msg.guild) return;
		if (msg.author.id === discordClient.user.id) return;
		const text: string = msg.content.trim();

		if (
            text.includes(tagPrefix)
            || msg.mentions.users.find((user: any) => user.username === discordClient.user.username)
            || Math.random() < 0.18
        ) {
			msg.channel.sendTyping();
			const messages = await fetchMessageData(msg.channel);
			const formattedMessages = formatMessages(messages);
			const response = await generateResponse(formattedMessages);
			setTimeout(
				() => msg.reply(replaceEmojis(response)),
				Math.floor(Math.random() * (2000 - 1000 + 1)) + 1000,
			);
		}
	} catch (e) {
		console.log(e);
	}
});
