from agent.planner import plan
import json

sub_questions = plan("How did Apple's and Microsoft's approaches to AI differ?")
for sq in sub_questions:
    print(json.dumps(sq))
    print('ticker repr:', repr(sq.get('ticker')))
    print('fiscal_year repr:', repr(sq.get('fiscal_year')))