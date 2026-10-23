# 🏛️ Antigravity Master Orchestrator: 60-Minute Video Engine

Этот документ является верховным регламентом Antigravity для производства 60-минутных псевдодокументальных расследований и атмосферного саспенса без использования внешних платных LLM API.

---

## 1. Базовый Производственный Стандарт
* **Формат:** 60 минут (3600 секунд) = ровно **8 глав по 7,5 минут**.
* **Сценарный объём:** 50–65 предложений (~850–1000 слов) на главу, суммарно ~7500–8000 слов на проект.
* **Сеттинг:** Обязательная опора на Библии мира из `ideas/bibles/` (топокарты, ГОСТы, приборы, физика).

---

## 2. Карта Этапов и Вызов Специализированных Скиллов

Производство делится на 3 стадии и управляется через модульные навыки в `.agents/skills/`:

| Стадия | Задача | Ответственный Скилл | Выходной Артефакт |
| :--- | :--- | :--- | :--- |
| **Стадия 1** | Проектирование 8-главного скелета, крах гипотез, пространственная сетка | [`story-architect`](file:///.agents/skills/story-architect/SKILL.md) | `data/projects/<id>/meta/project_plan.json` |
| **Шлюз 1** | Аудит макро-плана на логику, топографию и отсутствие провисаний | [`anti-slop-auditor`](file:///.agents/skills/anti-slop-auditor/SKILL.md) | Статус `APPROVED` для плана |
| **Стадия 2** | Поглавный драфтинг текста (50–65 предложений), 3 голоса, TTS-нормализация | [`narration-drafter`](file:///.agents/skills/narration-drafter/SKILL.md) | `data/projects/<id>/chapters/chXX/chapter_script.json` |
| **Аудио** | Разметка двухуровневого звука (амбиент -24 dB + акцентный Foley в паузах) | [`foley-sound-designer`](file:///.agents/skills/foley-sound-designer/SKILL.md) | Поля `ambient_bed` и `foley_cues` в JSON |
| **Шлюз 2** | Цензура слопа, проверка хронометража и запретительного лексикона главы | [`anti-slop-auditor`](file:///.agents/skills/anti-slop-auditor/SKILL.md) | Статус `APPROVED` для главы |

---

## 3. Каталог Настроек и Голосов
* **Голоса (TTS):** 3 эталонных сэмпла в `config/voices/` (`narrator.wav`, `engineer.wav`, `dispatcher.wav`).
* **Звуки (Foley):** Сэмплы в `assets/sfx/` (`ambient/`, `foley/`, `subterranean/`).
* **Визуал:** Pexels / Pixabay API + AI Image Harvester (Pollinations.ai) для редких аномалий.

---

## 4. Передача в Автономный Рендеринг (Стадия 3+)
Когда все 8 глав одобрены цензором и сохранены в файловой структуре, рендеринг видео запускается одной консольной командой без вызова внешних LLM:

```bash
py -3.10 src/pipeline.py --project-id <project_id> --resume-from voice
```
