from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import Command, CommandObject

import config as C
from retriever import load_or_rebuild_index, retrieve
from generator import generate_answer

MAIN_KB = ReplyKeyboardMarkup(
    keyboard=[[
        KeyboardButton(text="Объясни связь с DH"),
        KeyboardButton(text="Подбери научрука"),
    ],[
        KeyboardButton(text="Ресурсы программы"),
        KeyboardButton(text="Сузить тему ВКР"),
    ]],
    resize_keyboard=True
)

class VKRBot:
    def __init__(self):
        if not C.TELEGRAM_BOT_TOKEN:
            raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in environment")
        self.bot = Bot(token=C.TELEGRAM_BOT_TOKEN)
        self.dp = Dispatcher()
        self.store = None

        self.dp.message.register(self.cmd_start, Command("start"))
        self.dp.message.register(self.cmd_ask, Command("ask"))
        self.dp.message.register(self.cmd_supervisor, Command("supervisor"))
        self.dp.message.register(self.cmd_resources, Command("resources"))
        self.dp.message.register(self.cmd_refine, Command("refine"))

        # Button flows
        self.dp.message.register(self.btn_explain, F.text == "Объясни связь с DH")
        self.dp.message.register(self.btn_supervisor, F.text == "Подбери научрука")
        self.dp.message.register(self.btn_resources, F.text == "Ресурсы программы")
        self.dp.message.register(self.btn_refine, F.text == "Сузить тему ВКР")

    async def on_startup(self):
        self.store = load_or_rebuild_index()

    # Commands
    async def cmd_start(self, message: Message):
        await message.answer(
            "Привет! Я RAG-бот по ВКР. Я читаю PDF из двух папок Google Drive и отвечаю на вопросы на их основе.\n\n"
            "Команды:\n"
            "/ask <вопрос>\n"
            "/supervisor <тема>\n"
            "/resources [тема]\n"
            "/refine <черновик темы>\n\n"
            "Или используй кнопки ниже.",
            reply_markup=MAIN_KB
        )

    async def cmd_ask(self, message: Message, command: CommandObject):
        q = (command.args or "").strip()
        if not q:
            await message.answer("Напиши так: /ask твой вопрос")
            return
        await message.chat.do("typing")
        ctxs = retrieve(self.store, q, k=C.RETRIEVAL_K)
        ans = await generate_answer(q, ctxs)
        await message.answer(ans)

    async def cmd_supervisor(self, message: Message, command: CommandObject):
        topic = (command.args or "").strip()
        if not topic:
            await message.answer("Напиши так: /supervisor тема ВКР")
            return
        await message.chat.do("typing")
        q = f"Подбери научных руководителей по теме «{topic}», укажи основания и контакты из контекста."
        ctxs = retrieve(self.store, q, k=max(C.RETRIEVAL_K, 10))
        ans = await generate_answer(q, ctxs)
        await message.answer(ans)

    async def cmd_resources(self, message: Message, command: CommandObject):
        topic = (command.args or "").strip()
        await message.chat.do("typing")
        base = "Покажи релевантные ресурсы программы (курсы, лаборатории, доступы, корпуса данных) и шаги применения."
        q = f"{base} Учти тему: «{topic}»." if topic else base
        ctxs = retrieve(self.store, q, k=max(C.RETRIEVAL_K, 10))
        ans = await generate_answer(q, ctxs)
        await message.answer(ans)

    async def cmd_refine(self, message: Message, command: CommandObject):
        draft = (command.args or "").strip()
        if not draft:
            await message.answer("Напиши так: /refine черновик темы")
            return
        await message.chat.do("typing")
        q = f"Сузь и переформулируй тему на основе черновика «{draft}». Дай 3 варианта и методологию."
        ctxs = retrieve(self.store, q, k=max(C.RETRIEVAL_K, 10))
        ans = await generate_answer(q, ctxs)
        await message.answer(ans)

    # Buttons
    async def btn_explain(self, message: Message):
        await message.answer("Напиши тему ВКР одной фразой:")
        self.dp.message.register(self.capture_explain_topic, F.text)

    async def capture_explain_topic(self, message: Message):
        topic = message.text.strip()
        await message.chat.do("typing")
        q = f"Объясни, как тема «{topic}» соотносится с Digital Humanities. Дай методы и примеры из контекста."
        ctxs = retrieve(self.store, q, k=max(C.RETRIEVAL_K, 10))
        ans = await generate_answer(q, ctxs)
        await message.answer(ans)

    async def btn_supervisor(self, message: Message):
        await message.answer("Напиши тему ВКР для подбора научрука:")
        self.dp.message.register(self.capture_supervisor_topic, F.text)

    async def capture_supervisor_topic(self, message: Message):
        topic = message.text.strip()
        await message.chat.do("typing")
        q = f"Подбери научных руководителей по теме «{topic}», укажи основания и контакты из контекста."
        ctxs = retrieve(self.store, q, k=max(C.RETRIEVAL_K, 12))
        ans = await generate_answer(q, ctxs)
        await message.answer(ans)

    async def btn_resources(self, message: Message):
        await message.answer("Если хочешь, укажи тему ВКР (или просто отправь пустое сообщение):")
        self.dp.message.register(self.capture_resources_topic, F.text)

    async def capture_resources_topic(self, message: Message):
        topic = message.text.strip()
        topic = topic if topic else None
        await message.chat.do("typing")
        base = "Покажи релевантные ресурсы программы (курсы, лаборатории, доступы, корпуса данных) и шаги применения."
        q = f"{base} Учти тему: «{topic}»." if topic else base
        ctxs = retrieve(self.store, q, k=max(C.RETRIEVAL_K, 10))
        ans = await generate_answer(q, ctxs)
        await message.answer(ans)

    async def btn_refine(self, message: Message):
        await message.answer("Пришли черновик формулировки:")
        self.dp.message.register(self.capture_refine_draft, F.text)

    async def capture_refine_draft(self, message: Message):
        draft = message.text.strip()
        await message.chat.do("typing")
        q = f"Сузь и переформулируй тему на основе черновика «{draft}». Дай 3 варианта и методологию."
        ctxs = retrieve(self.store, q, k=max(C.RETRIEVAL_K, 10))
        ans = await generate_answer(q, ctxs)
        await message.answer(ans)
