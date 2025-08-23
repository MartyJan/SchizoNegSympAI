CAINS_ITEM_NAME = [
    "親密關係的動機（家人／配偶／伴侶）",
    "親密關係的動機（友誼／浪漫）",
    "過去一週愉快的社交活動的頻率",
    "預期下週愉快的社交活動的頻率",
    "工作或上學的動機",
    "在下週工作／上學相關的活動中預期的愉快或愉快體驗的頻率",
    "休閒活動的動機",
    "過去一週經歷到愉快的休閒活動",
    "在下週的嗜好或休閒活動中預期的愉快或愉快經驗的頻率",
]


ITEM_ABBR_TO_NAME = {
    "C1": CAINS_ITEM_NAME[0],
    "C2": CAINS_ITEM_NAME[1],
    "C3": CAINS_ITEM_NAME[2],
    "C4": CAINS_ITEM_NAME[3],
    "C5": CAINS_ITEM_NAME[4],
    "C6": CAINS_ITEM_NAME[5],
    "C7": CAINS_ITEM_NAME[6],
    "C8": CAINS_ITEM_NAME[7],
    "C9": CAINS_ITEM_NAME[8],
}

ITEM_NAME_TO_ABBR = {
    CAINS_ITEM_NAME[0]: "C1",
    CAINS_ITEM_NAME[1]: "C2",
    CAINS_ITEM_NAME[2]: "C3",
    CAINS_ITEM_NAME[3]: "C4",
    CAINS_ITEM_NAME[4]: "C5",
    CAINS_ITEM_NAME[5]: "C6",
    CAINS_ITEM_NAME[6]: "C7",
    CAINS_ITEM_NAME[7]: "C8",
    CAINS_ITEM_NAME[8]: "C9",
}

ITEM_TYPES_TO_ABBR_LIST = {
    "C1-9": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
    "C1-4": ["C1", "C2", "C3", "C4"],
    "C5-6": ["C5", "C6"],
    "C7-9": ["C7", "C8", "C9"],
}
