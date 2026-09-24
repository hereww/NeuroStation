// NeuroStation overlay: minimal, upstream-friendly internationalization layer.
// Locale may be passed with -Dopenbci.locale=zh-CN or NEUROSTATION_LANGUAGE.

String workstationLocale = "zh-CN";
JSONObject workstationEnglish;
JSONObject workstationChinese;

void setupWorkstationI18n() {
    String requested = System.getProperty("openbci.locale");
    if (requested == null || requested.length() == 0) {
        requested = System.getenv("NEUROSTATION_LANGUAGE");
    }
    if (requested != null && requested.toLowerCase().startsWith("en")) {
        workstationLocale = "en-US";
    }
    workstationEnglish = loadJSONObject("workstation-i18n/en-US.json");
    workstationChinese = loadJSONObject("workstation-i18n/zh-CN.json");
}

boolean workstationUsesChinese() {
    return workstationLocale.equals("zh-CN");
}

String tr(String key) {
    JSONObject selected = workstationUsesChinese() ? workstationChinese : workstationEnglish;
    if (selected != null && selected.hasKey(key)) {
        return selected.getString(key);
    }
    if (workstationEnglish != null && workstationEnglish.hasKey(key)) {
        return workstationEnglish.getString(key);
    }
    return key;
}

String workstationUIFont(String upstreamFont) {
    if (!workstationUsesChinese()) {
        return upstreamFont;
    }
    if (isWindows()) {
        return "Microsoft YaHei UI";
    }
    if (isMac()) {
        return "PingFang SC";
    }
    return "Noto Sans CJK SC";
}
