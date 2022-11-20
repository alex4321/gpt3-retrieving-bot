from completion_reactions import split_completion, Command


def test_split_completion():
    result = split_completion("Search: China GDP per capita 2019 Search: US GDP per capita 2019 Answer: US")
    assert result == [
        Command("Search", "China GDP per capita 2019"),
        Command("Search", "US GDP per capita 2019"),
        Command("Answer", "US"),
    ]
