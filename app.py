import streamlit as st
import pandas as pd
from collections import defaultdict, deque

# --------------------------
# Grammar Parsing and Processing
# --------------------------

def parse_grammar(grammar_input):
    """Parse the input grammar into a dictionary."""
    grammar = {}
    lines = grammar_input.strip().split('\n')
    for line in lines:
        if '->' not in line:
            continue
        lhs, rhs = line.split('->', 1)
        lhs = lhs.strip()
        productions = [p.strip() for p in rhs.split('|')]
        if lhs not in grammar:
            grammar[lhs] = []
        grammar[lhs].extend(productions)
    return grammar

def process_grammar(grammar):
    """
    Extract terminals, non-terminals, and create augmented grammar.
    Returns: start_symbol, terminals, non_terminals, augmented_grammar.
    """
    start_symbol = list(grammar.keys())[0]
    augmented_start = f"{start_symbol}'"
    augmented_grammar = [(augmented_start, (start_symbol,))]
    
    non_terminals = set(grammar.keys())
    terminals = set()
    
    for nt, productions in grammar.items():
        for prod in productions:
            symbols = prod.split()
            for symbol in symbols:
                if symbol not in non_terminals and symbol != '#':
                    terminals.add(symbol)
            augmented_grammar.append((nt, tuple(symbols) if prod != '#' else tuple()))
    terminals.add('$')
    return start_symbol, terminals, non_terminals, augmented_grammar

# --------------------------
# FIRST and FOLLOW Sets
# --------------------------

def compute_first_sets(grammar, terminals, non_terminals):
    """Compute FIRST sets for all symbols."""
    first_sets = {}
    for nt in non_terminals:
        first_sets[nt] = set()
    for terminal in terminals:
        first_sets[terminal] = {terminal}
    
    for nt, productions in grammar.items():
        if '#' in productions:
            first_sets[nt].add('#')
    
    changed = True
    while changed:
        changed = False
        for nt, productions in grammar.items():
            for prod in productions:
                if prod == '#':
                    continue
                symbols = prod.split()
                i = 0
                can_derive_epsilon = True
                while i < len(symbols) and can_derive_epsilon:
                    symbol = symbols[i]
                    if symbol in first_sets:
                        non_epsilon = first_sets[symbol] - {'#'}
                        old_len = len(first_sets[nt])
                        first_sets[nt].update(non_epsilon)
                        if len(first_sets[nt]) > old_len:
                            changed = True
                    can_derive_epsilon = ('#' in first_sets.get(symbol, set()))
                    i += 1
                if can_derive_epsilon and i == len(symbols) and '#' not in first_sets[nt]:
                    first_sets[nt].add('#')
                    changed = True
    return first_sets

def compute_follow_sets(grammar, non_terminals, start_symbol, first_sets):
    """Compute FOLLOW sets for all non-terminals."""
    follow_sets = {nt: set() for nt in non_terminals}
    follow_sets[start_symbol].add('$')
    
    changed = True
    while changed:
        changed = False
        for nt, productions in grammar.items():
            for prod in productions:
                if prod == '#':
                    continue
                symbols = prod.split()
                for i, symbol in enumerate(symbols):
                    if symbol in non_terminals:
                        if i < len(symbols) - 1:
                            next_seq = symbols[i+1:]
                            seq_first = compute_first_of_sequence(next_seq, first_sets)
                            non_epsilon = seq_first - {'#'}
                            old_len = len(follow_sets[symbol])
                            follow_sets[symbol].update(non_epsilon)
                            if len(follow_sets[symbol]) > old_len:
                                changed = True
                            if '#' in seq_first:
                                old_len = len(follow_sets[symbol])
                                follow_sets[symbol].update(follow_sets[nt])
                                if len(follow_sets[symbol]) > old_len:
                                    changed = True
                        else:
                            old_len = len(follow_sets[symbol])
                            follow_sets[symbol].update(follow_sets[nt])
                            if len(follow_sets[symbol]) > old_len:
                                changed = True
    return follow_sets

def compute_first_of_sequence(symbols, first_sets):
    """Compute FIRST set of a sequence of symbols."""
    if not symbols:
        return {'#'}
    result = set()
    all_epsilon = True
    for symbol in symbols:
        non_epsilon = first_sets[symbol] - {'#'}
        result.update(non_epsilon)
        if '#' not in first_sets[symbol]:
            all_epsilon = False
            break
    if all_epsilon:
        result.add('#')
    return result

# --------------------------
# LR(0) Items, Closure, and Goto
# --------------------------

def closure(items, grammar, non_terminals):
    """Compute the closure of a set of LR(0) items.
       Each item is a tuple (LHS, rhs-tuple) with 'DOT' marking the position.
    """
    result = set(items)
    changed = True
    while changed:
        changed = False
        new_items = set()
        for item in result:
            nt, rhs = item
            try:
                dot_pos = rhs.index('DOT')
                if dot_pos < len(rhs) - 1:
                    next_symbol = rhs[dot_pos + 1]
                    if next_symbol in non_terminals:
                        for prod in grammar.get(next_symbol, []):
                            if prod == '#':
                                new_item = (next_symbol, ('DOT',))
                            else:
                                new_item = (next_symbol, tuple(['DOT'] + prod.split()))
                            if new_item not in result:
                                new_items.add(new_item)
                                changed = True
            except ValueError:
                continue
        result.update(new_items)
    return frozenset(result)

def goto(items, symbol, grammar, non_terminals):
    """Compute goto(I, X) where I is a set of items and X is a grammar symbol."""
    next_items = set()
    for item in items:
        nt, rhs = item
        try:
            dot_pos = rhs.index('DOT')
            if dot_pos < len(rhs) - 1 and rhs[dot_pos + 1] == symbol:
                new_rhs = list(rhs)
                new_rhs[dot_pos], new_rhs[dot_pos + 1] = new_rhs[dot_pos + 1], new_rhs[dot_pos]
                next_items.add((nt, tuple(new_rhs)))
        except ValueError:
            continue
    if next_items:
        return closure(next_items, grammar, non_terminals)
    else:
        return None

def build_canonical_collection(grammar, non_terminals, start_symbol):
    """Build the canonical collection of LR(0) items and the goto table."""
    augmented_start = f"{start_symbol}'"
    initial_item = (augmented_start, ('DOT', start_symbol))
    initial_state = closure({initial_item}, grammar, non_terminals)
    
    canonical_collection = [initial_state]
    goto_table = {}
    
    i = 0
    while i < len(canonical_collection):
        state = canonical_collection[i]
        next_symbols = set()
        for item in state:
            nt, rhs = item
            try:
                dot_pos = rhs.index('DOT')
                if dot_pos < len(rhs) - 1:
                    next_symbols.add(rhs[dot_pos + 1])
            except ValueError:
                continue
        for symbol in next_symbols:
            next_state = goto(state, symbol, grammar, non_terminals)
            if next_state and next_state not in canonical_collection:
                canonical_collection.append(next_state)
            if next_state:
                goto_table[(i, symbol)] = canonical_collection.index(next_state)
        i += 1
    return canonical_collection, goto_table

# --------------------------
# Parsing Table Construction
# --------------------------

def construct_parsing_table(canonical_collection, goto_table, terminals, non_terminals, augmented_grammar, grammar, follow_sets, start_symbol):
    """Construct the SLR parsing table."""
    parsing_table = {}
    for i in range(len(canonical_collection)):
        parsing_table[i] = {}
        for symbol in terminals | non_terminals:
            parsing_table[i][symbol] = []
    
    for i, state in enumerate(canonical_collection):
        for item in state:
            nt, rhs = item
            try:
                dot_pos = rhs.index('DOT')
                if dot_pos < len(rhs) - 1:
                    next_symbol = rhs[dot_pos + 1]
                    if next_symbol in terminals and (i, next_symbol) in goto_table:
                        next_state = goto_table[(i, next_symbol)]
                        action = f"s{next_state}"
                        if action not in parsing_table[i][next_symbol]:
                            parsing_table[i][next_symbol].append(action)

                elif dot_pos == len(rhs) - 1 or (len(rhs) == 1 and rhs[0] == 'DOT'):
                    if nt == f"{start_symbol}'" and len(rhs) == 2 and rhs[1] == start_symbol:
                        parsing_table[i]['$'].append("acc")
                    else:
                        prod_num = -1
                        item_rhs = tuple(s for s in rhs if s != 'DOT')
                        for j, (prod_nt, prod_rhs) in enumerate(augmented_grammar):
                            if prod_nt == nt and prod_rhs == item_rhs:
                                prod_num = j
                                break
                        if prod_num != -1:
                            for follow_symbol in follow_sets.get(nt, set()):
                                action = f"r{prod_num}"
                                if action not in parsing_table[i][follow_symbol]:
                                    parsing_table[i][follow_symbol].append(action)
            except ValueError:
                continue
        for nt in non_terminals:
            if (i, nt) in goto_table:
                next_state = goto_table[(i, nt)]
                parsing_table[i][nt].append(str(next_state))
    return parsing_table

# --------------------------
# Parsing Simulation
# --------------------------

def parse_input(input_string, parsing_table, augmented_grammar, canonical_collection):
    """Parse the input string using the SLR parsing table.
       Returns (list of step actions, success flag).
    """
    tokens = input_string.strip().split()
    tokens.append('$')
    stack = [0]
    pointer = 0
    actions = []
    
    while True:
        current_state = stack[-1]
        current_token = tokens[pointer]
        if current_token not in parsing_table[current_state]:
            actions.append(f"Error: Unexpected token {current_token}")
            return actions, False
        possible_actions = parsing_table[current_state][current_token]
        if not possible_actions:
            actions.append(f"Error: No action for state {current_state} and token {current_token}")
            return actions, False
        if len(possible_actions) > 1:
            actions.append(f"Error: Conflict in parsing table for state {current_state} and token {current_token}")
            return actions, False
        action = possible_actions[0]
        if action.startswith('s'):
            next_state = int(action[1:])
            stack.append(current_token)
            stack.append(next_state)
            actions.append(f"Shift {current_token}, go to state {next_state}")
            pointer += 1
        elif action.startswith('r'):
            prod_num = int(action[1:])
            prod_nt, prod_rhs = augmented_grammar[prod_num]
            for _ in range(2 * len(prod_rhs)):
                stack.pop()
            current_state = stack[-1]
            stack.append(prod_nt)
            if prod_nt in parsing_table[current_state]:
                goto_actions = parsing_table[current_state][prod_nt]
                if goto_actions:
                    next_state = int(goto_actions[0])
                    stack.append(next_state)
                    rhs_str = ' '.join(prod_rhs) if prod_rhs else '#'
                    actions.append(f"Reduce by {prod_nt} -> {rhs_str}, go to state {next_state}")
                else:
                    actions.append(f"Error: No goto action for state {current_state} and non-terminal {prod_nt}")
                    return actions, False
            else:
                actions.append(f"Error: No entry in parsing table for state {current_state} and non-terminal {prod_nt}")
                return actions, False
        elif action == "acc":
            actions.append("Accept! Input successfully parsed.")
            return actions, True
        else:
            actions.append(f"Error: Invalid action {action}")
            return actions, False

# --------------------------
# Formatting Functions for Display
# --------------------------

def format_parsing_table(parsing_table, canonical_collection, terminals, non_terminals):
    """Format the parsing table into a DataFrame (without an extra 'State' column)."""
    sorted_terminals = sorted(terminals)
    sorted_non_terminals = sorted(non_terminals)
    header = sorted_terminals + sorted_non_terminals
    rows = {}
    for state in range(len(canonical_collection)):
        row = []
        for terminal in sorted_terminals:
            acts = parsing_table[state].get(terminal, [])
            row.append('/'.join(acts) if acts else '')
        for nt in sorted_non_terminals:
            acts = parsing_table[state].get(nt, [])
            row.append('/'.join(acts) if acts else '')
        rows[state] = row
    return pd.DataFrame.from_dict(rows, orient='index', columns=header)

def get_productions(augmented_grammar):
    """Return a list of productions (excluding the augmented production) as strings."""
    productions = []
    for i, (nt, rhs) in enumerate(augmented_grammar):
        if i == 0:
            continue
        prod_str = f"{i}. {nt} -> {' '.join(rhs) if rhs else '#'}"
        productions.append(prod_str)
    return productions

def format_lr_items(canonical_collection):
    """Format LR(0) items for display with derivation order."""
    formatted = []
    for i, state in enumerate(canonical_collection):
        state_items = []
        for nt, rhs in state:
            formatted_rhs = ['•' if symbol == 'DOT' else symbol for symbol in rhs]
            state_items.append(f"{nt} -> {' '.join(formatted_rhs)}")
        formatted.append((i, state_items))
    return formatted

# --------------------------
# Main Function (Enhanced Streamlit UI)
# --------------------------

def main():
    st.set_page_config(page_title="SLR Parser Generator", layout="wide")
    st.markdown("[Source Code](https://github.com/singh-sudhir16/SLR-parse-table-generator)")
    st.title("SLR Parser Generator")
    
    st.sidebar.header("Grammar Input")
    example_grammar = """
E -> E + T | T
T -> T * F | F
F -> ( E ) | id
    """
    grammar_input = st.sidebar.text_area("Enter grammar (one production per line, use '#' for epsilon):", 
                                           value=example_grammar, height=200)
    
    if st.sidebar.button("Generate Parser"):
        try:
            # Processing
            grammar = parse_grammar(grammar_input)
            start_symbol, terminals, non_terminals, augmented_grammar = process_grammar(grammar)
            first_sets = compute_first_sets(grammar, terminals, non_terminals)
            follow_sets = compute_follow_sets(grammar, non_terminals, start_symbol, first_sets)
            canonical_collection, goto_table = build_canonical_collection(grammar, non_terminals, start_symbol)
            parsing_table = construct_parsing_table(canonical_collection, goto_table, terminals, non_terminals, 
                                                     augmented_grammar, grammar, follow_sets, start_symbol)
            
            tab1, tab2, tab3, tab4 = st.tabs(["Grammar Analysis", "Productions", "LR(0) Items", "Parsing Table"])
            
            with tab1:
                st.header("Grammar Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Terminals")
                    st.write(", ".join(sorted(terminals)))
                with col2:
                    st.subheader("Non-terminals")
                    st.write(", ".join(sorted(non_terminals)))
                
                with st.expander("FIRST Sets"):
                    first_df = pd.DataFrame({
                        'Non-terminal': sorted(non_terminals),
                        'FIRST Set': [", ".join(sorted(first_sets.get(nt, []))) for nt in sorted(non_terminals)]
                    })
                    st.dataframe(first_df)
                
                with st.expander("FOLLOW Sets"):
                    follow_df = pd.DataFrame({
                        'Non-terminal': sorted(non_terminals),
                        'FOLLOW Set': [", ".join(sorted(follow_sets.get(nt, []))) for nt in sorted(non_terminals)]
                    })
                    st.dataframe(follow_df)
            
            with tab2:
                st.header("Productions")
                for prod in get_productions(augmented_grammar):
                    st.write(prod)
            
            with tab3:
                st.header("Canonical Collection of LR(0) Items")
                for i, items in format_lr_items(canonical_collection):
                    st.write(f"**State {i}:**")
                    for item in items:
                        st.write(f"- {item}")
            
            with tab4:
                st.header("SLR Parsing Table")
                parsing_table_df = format_parsing_table(parsing_table, canonical_collection, terminals, non_terminals)
                st.dataframe(parsing_table_df)
            
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
import streamlit as st



if __name__ == "__main__":
    main()
