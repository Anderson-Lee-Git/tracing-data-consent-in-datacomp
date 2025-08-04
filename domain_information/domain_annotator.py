import streamlit as st
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv()

# Available users
USERS = ["Anderson", "Willie", "Rachel"]

# Get environment variables with proper None handling
annotations_dir = os.getenv("DOMAIN_ANNOTATION_OUTPUT_DIR")
domain_set_dir = os.getenv("DOMAIN_ANNOTATION_CANDIDATE_DIR")
metadata_dir = os.getenv("SMALL_ENGLISH_FILTER_ANNOTATED_METADATA_DIR")

if not annotations_dir or not domain_set_dir or not metadata_dir:
    raise ValueError("Required environment variables are not set")

ANNOTATIONS_DIR = Path(annotations_dir)
DOMAIN_SET_TO_ANNOTATE_DIR = Path(domain_set_dir)
SCHEMA_FILE = ANNOTATIONS_DIR / "annotation_schema.json"
BASE_TO_FULL_DOMAIN_MAPPING = json.load(
    open(
        Path(metadata_dir) / "base_to_full_domain_mapping.json",
        "r",
    )
)


class FieldType(str, Enum):
    TEXT = "Text"
    BOOLEAN = "Boolean"
    SINGLE_SELECT = "Single Select"
    MULTI_SELECT = "Multi Select"
    TEXTAREA = "Text Area"


class FieldSchema(BaseModel):
    id: str
    name: str
    type: FieldType
    options: Optional[List[str]] = None
    description: Optional[str] = None
    required: bool = False


class AnnotationSchema(BaseModel):
    fields: List[FieldSchema]


FIELD_TYPE_ORDER = [
    FieldType.BOOLEAN,
    FieldType.SINGLE_SELECT,
    FieldType.MULTI_SELECT,
    FieldType.TEXT,
    FieldType.TEXTAREA,
]

# Initialize default schema with current fields
DEFAULT_SCHEMA = AnnotationSchema(
    fields=[
        FieldSchema(
            id="tos_links",
            name="Terms of Service Links",
            type=FieldType.TEXTAREA,
            description="Terms of Service Links (one per line)",
            required=True,
        ),
        FieldSchema(
            id="notes",
            name="Notes",
            type=FieldType.TEXTAREA,
            required=False,
        ),
    ]
)


def field_sort_key(field):
    try:
        return FIELD_TYPE_ORDER.index(field.type)
    except ValueError:
        return len(FIELD_TYPE_ORDER)


def order_fields(fields: List[FieldSchema]) -> List[FieldSchema]:
    evidence_fields = dict(
        [(field.id, field) for field in fields if field.id.endswith("evidence")]
    )
    other_fields = [field for field in fields if not field.id.endswith("evidence")]
    sorted_fields = sorted(other_fields, key=field_sort_key)
    # append evidence feilds to each field in the sorted field based on the prefix
    complete_fields = []
    for field in sorted_fields:
        complete_fields.append(field)
        if f"{field.id}_evidence" in evidence_fields:
            complete_fields.append(evidence_fields[f"{field.id}_evidence"])
    return complete_fields


def load_annotation_schema() -> AnnotationSchema:
    """Load the annotation schema or create default if not exists"""
    if not SCHEMA_FILE.exists():
        schema = DEFAULT_SCHEMA
        with open(SCHEMA_FILE, "w") as f:
            f.write(schema.model_dump_json(indent=4))
        return schema

    with open(SCHEMA_FILE, "r") as f:
        schema_data = json.load(f)
    return AnnotationSchema(**schema_data)


def save_annotation_schema(schema: AnnotationSchema):
    """Save the annotation schema"""
    with open(SCHEMA_FILE, "w") as f:
        f.write(schema.model_dump_json(indent=4))


def load_domains(file_path: str) -> List[str]:
    """Load domains from a JSON file"""
    with open(file_path, "r") as f:
        return json.load(f)


def load_union_annotations(user: str) -> Dict[str, Dict[str, Any]]:
    """Load existing annotations for a specific user"""
    union_file = ANNOTATIONS_DIR / f"{user}_union_annotations.json"
    if not union_file.exists():
        return {}
    with open(union_file, "r") as f:
        return json.load(f)


def save_annotation(
    domain: str,
    annotation_data: Dict[str, Any],
    domain_set_to_annotate: Path,
    user: str,
):
    """Save an annotation to both set-specific and union files for a specific user"""
    # Save to set-specific annotations
    st.write(f"Saving annotation for {domain}")
    annotations_file = (
        ANNOTATIONS_DIR / f"{user}_{domain_set_to_annotate.stem}_annotations.json"
    )

    if annotations_file.exists():
        with open(annotations_file, "r") as f:
            annotations = json.load(f)
    else:
        annotations = {}
    st.write(annotation_data)
    schema = load_annotation_schema()
    annotation_data = {
        k: v
        for k, v in sorted(
            annotation_data.items(),
            key=lambda x: x[0],
        )
        if k in [field.id for field in schema.fields]
    }
    annotations[domain] = {
        "domain": domain,
        **annotation_data,
    }

    with open(annotations_file, "w") as f:
        json.dump(annotations, f, indent=4)

    # Save to union annotations
    union_annotations = load_union_annotations(user)
    union_annotations[domain] = {
        "domain": domain,
        **annotation_data,
    }

    with open(ANNOTATIONS_DIR / f"{user}_union_annotations.json", "w") as f:
        json.dump(union_annotations, f, indent=4)


def render_tos_text(domain: str, tos_text: Dict[str, str]):
    """Render ToS text if available"""
    st.write("## Terms of Service Text")
    for tos_link, text in tos_text.items():
        with st.expander(f"ToS from {tos_link}"):
            st.write(text)


def schema_editor():
    """UI for editing the annotation schema"""
    st.header("Schema Editor")

    # Force load the schema even if the tab is not active
    schema = load_annotation_schema()

    # Show existing fields
    st.subheader("Current Fields")
    for i, field in enumerate(schema.fields):
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.markdown(f"**{field.name}** (`{field.type}`)")
        with col2:
            if field.options:
                st.write(f"Options: {', '.join(field.options)}")
        with col3:
            sub_col1, sub_col2 = st.columns([1, 1])
            with sub_col1:
                if st.button("Edit", key=f"edit_{field.id}"):
                    st.session_state.editing_field = field.id
            with sub_col2:
                if st.button("Delete", key=f"delete_{field.id}"):
                    schema.fields = [f for f in schema.fields if f.id != field.id]
                    save_annotation_schema(schema)
                    st.rerun()

    # Add new field button
    if st.button("Add New Field"):
        st.session_state.editing_field = "new"

    # Field editor
    if "editing_field" in st.session_state:
        st.subheader("Field Editor")

        if st.session_state.editing_field == "new":
            # New field
            field_id = st.text_input("Field ID")
            field_name = st.text_input("Field Name")
            field_type = st.selectbox(
                "Field Type", options=[t.value for t in FieldType]
            )
            field_description = st.text_area("Description (optional)")
            field_required = st.checkbox("Required")

            field_options = None
            if field_type in [FieldType.SINGLE_SELECT, FieldType.MULTI_SELECT]:
                options_text = st.text_area("Options (one per line)")
                if options_text:
                    field_options = [
                        opt.strip() for opt in options_text.strip().split("\n")
                    ]

            if st.button("Save Field"):
                if field_name:
                    new_field = FieldSchema(
                        id=field_id,
                        name=field_name,
                        type=FieldType(field_type),
                        options=field_options,
                        description=field_description or None,
                        required=field_required,
                    )
                    schema.fields.append(new_field)
                    save_annotation_schema(schema)
                    del st.session_state.editing_field
                    st.rerun()
                else:
                    st.error("Field name is required")
        else:
            # Edit existing field
            field = next(
                (f for f in schema.fields if f.id == st.session_state.editing_field),
                None,
            )
            if field:
                field_name = st.text_input("Field Name", value=field.name)
                field_type = st.selectbox(
                    "Field Type",
                    options=[t.value for t in FieldType],
                    index=[t.value for t in FieldType].index(field.type),
                )
                field_description = st.text_area(
                    "Description (optional)", value=field.description or ""
                )
                field_required = st.checkbox("Required", value=field.required)

                field_options = None
                if field_type in [FieldType.SINGLE_SELECT, FieldType.MULTI_SELECT]:
                    options_text = st.text_area(
                        "Options (one per line)", value="\n".join(field.options or [])
                    )
                    if options_text:
                        field_options = [
                            opt.strip() for opt in options_text.strip().split("\n")
                        ]

                if st.button("Update Field"):
                    if field_name:
                        for i, f in enumerate(schema.fields):
                            if f.id == field.id:
                                schema.fields[i] = FieldSchema(
                                    id=field.id,
                                    name=field_name,
                                    type=FieldType(field_type),
                                    options=field_options,
                                    description=field_description or None,
                                    required=field_required,
                                )
                                break
                        save_annotation_schema(schema)
                        del st.session_state.editing_field
                        st.rerun()
                    else:
                        st.error("Field name is required")

        if st.button("Cancel"):
            del st.session_state.editing_field
            st.rerun()


def render_field(field: FieldSchema, value: Any) -> Any:
    """Render a form field based on schema and return the value"""
    if field.type == FieldType.TEXT:
        return st.text_input(field.name, value=value or "", help=field.description)

    elif field.type == FieldType.TEXTAREA:
        if isinstance(value, list):
            # Handle special case for tos_links which was a list
            value = "\n".join(value)
        return st.text_area(field.name, value=value or "", help=field.description)

    elif field.type == FieldType.BOOLEAN:
        display_value = value if value is not None else False
        return st.checkbox(field.name, value=display_value, help=field.description)

    elif field.type == FieldType.SINGLE_SELECT:
        options = field.options or []
        index = 0
        if value in options:
            index = options.index(value)
        return st.selectbox(
            field.name,
            options=options,
            index=index,
            help=field.description,
        )

    elif field.type == FieldType.MULTI_SELECT:
        options = field.options or []
        default = []
        if value:
            default = [opt for opt in value if opt in options]
        return st.multiselect(
            field.name,
            options=options,
            default=default,
            help=field.description,
        )


def process_field_value(field: FieldSchema, value: Any) -> Any:
    """Process a field value for storage"""
    if field.type == FieldType.TEXTAREA and field.id == "tos_links":
        # Special case for tos_links
        if isinstance(value, str):
            return value.strip().split("\n") if value.strip() else []
    return value


def clear_state():
    st.session_state.clear()


def main():
    st.set_page_config(layout="wide", page_title="Domain Annotator")

    # Initialize session state
    if "tab" not in st.session_state:
        st.session_state.tab = "annotate"
    if "current_user" not in st.session_state:
        st.session_state.current_user = USERS[0]

    # User selection
    st.sidebar.header("User Selection")
    selected_user = st.sidebar.selectbox(
        "Select User",
        options=USERS,
        index=USERS.index(st.session_state.current_user),
        key="user_selector",
    )

    # Update session state if user changed
    if selected_user != st.session_state.current_user:
        st.session_state.current_user = selected_user
        # Clear domain-specific state when user changes
        if "current_domain_idx" in st.session_state:
            del st.session_state.current_domain_idx
        if "preload_domain" in st.session_state:
            del st.session_state.preload_domain
        st.rerun()

    # Tab navigation
    tab1, tab2, tab3 = st.tabs(["Annotate Domains", "Edit Schema", "View Annotations"])

    with tab1:
        domain_annotator()

    with tab2:
        schema_editor()

    with tab3:
        annotation_viewer()

    # Track active tab
    if tab1.active:
        st.session_state.tab = "annotate"
    elif tab2.active:
        st.session_state.tab = "edit_schema"
    elif tab3.active:
        st.session_state.tab = "view_annotations"


def domain_annotator():
    st.title("Domain Annotation Tool")

    union_annotations = load_union_annotations(st.session_state.current_user)

    col1, col2 = st.columns([1, 3])

    with col1:
        # Domain set selection
        domain_sets_options = list(DOMAIN_SET_TO_ANNOTATE_DIR.glob("*.json"))
        domain_set_to_annotate = st.selectbox(
            "Select domain set",
            options=domain_sets_options,
            format_func=lambda x: x.stem,
            on_change=clear_state,
        )

        # Load domains
        domains = load_domains(str(domain_set_to_annotate))

        # Initialize current index
        if "current_domain_idx" not in st.session_state:
            st.session_state.current_domain_idx = 0

        current_domain = domains[st.session_state.current_domain_idx]

        # Initialize preload_domain in session state if not exists
        if "preload_domain" not in st.session_state:
            st.session_state.preload_domain = current_domain

        # Navigation
        st.write("### Navigation")

        progress_col1, progress_col2 = st.columns([1, 2])
        with progress_col1:
            st.metric("Current", st.session_state.current_domain_idx + 1)
        with progress_col2:
            st.progress(
                min(1.0, (st.session_state.current_domain_idx + 1) / len(domains))
            )

        st.write(f"Total: {len(domains)} domains")

        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button(
                "← Previous",
                use_container_width=True,
                disabled=st.session_state.current_domain_idx <= 0,
            ):
                st.session_state.current_domain_idx = max(
                    0, st.session_state.current_domain_idx - 1
                )
                # Reset preload_domain when navigating
                st.session_state.preload_domain = domains[
                    st.session_state.current_domain_idx
                ]
                st.rerun()
        with nav_col2:
            if st.button(
                "Next →",
                use_container_width=True,
                disabled=st.session_state.current_domain_idx >= len(domains) - 1,
            ):
                st.session_state.current_domain_idx = min(
                    len(domains) - 1, st.session_state.current_domain_idx + 1
                )
                # Reset preload_domain when navigating
                st.session_state.preload_domain = domains[
                    st.session_state.current_domain_idx
                ]
                st.rerun()

        # Jump to domain
        st.write("### Jump to Domain")
        jump_idx = st.number_input(
            "Domain #",
            min_value=1,
            max_value=len(domains),
            value=st.session_state.current_domain_idx + 1,
        )
        if st.button("Go", use_container_width=True):
            st.session_state.current_domain_idx = jump_idx - 1
            # Reset preload_domain when jumping
            st.session_state.preload_domain = domains[
                st.session_state.current_domain_idx
            ]
            st.rerun()

        # Add buttons for annotation loading
        st.write("### Load Annotation")
        selected_domain = st.selectbox(
            "Load annotation from domain:",
            options=list(union_annotations.keys()),
            key="load_from_domain",
        )
        if st.button(
            "Load Selected Domain's Annotation",
            use_container_width=True,
        ):
            st.session_state.preload_domain = selected_domain
            st.rerun()

    with col2:
        if st.session_state.current_domain_idx >= len(domains):
            st.success("All domains have been annotated!")
            return

        # Header with domain information
        st.header(f"Domain: {current_domain}")

        if current_domain in BASE_TO_FULL_DOMAIN_MAPPING:
            with st.expander("View full domain examples"):
                for full_domain in BASE_TO_FULL_DOMAIN_MAPPING[current_domain][:10]:
                    st.write(f"• {full_domain}")

        # Load schema and existing annotations
        schema = load_annotation_schema()

        # Initialize current_annotation from session state or empty dict
        if st.session_state.preload_domain in union_annotations:
            st.info(f"Preloaded annotation from {st.session_state.preload_domain}")
            current_annotation = union_annotations[st.session_state.preload_domain]
        else:
            current_annotation = {}

        # Annotation form
        with st.form("annotation_form"):
            st.write("### Domain Annotation")

            # Dynamic fields based on schema
            field_values = {}
            sorted_fields = order_fields(schema.fields)
            for field in sorted_fields:
                current_value = current_annotation.get(field.id, None)
                field_values[field.id] = render_field(field, current_value)

            # Submit button
            submitted = st.form_submit_button("Save Annotation")

        st.write(field_values)

        if submitted:
            # Save annotation
            # Process values for storage
            processed_values = {}
            for field in schema.fields:
                processed_values[field.id] = process_field_value(
                    field, field_values[field.id]
                )
            # Add domain to the annotation
            processed_values["domain"] = current_domain
            save_annotation(
                current_domain,
                processed_values,
                domain_set_to_annotate,
                st.session_state.current_user,
            )
            # Move to next domain
            st.session_state.current_domain_idx = (
                st.session_state.current_domain_idx + 1
            ) % len(domains)
            # Clear the preload after saving to avoid confusion
            st.session_state.preload_domain = domains[
                st.session_state.current_domain_idx
            ]
            st.rerun()

        # Display ToS text if available
        if "tos_text" not in st.session_state:
            tos_file = ANNOTATIONS_DIR / "tos_text.json"
            if tos_file.exists():
                with open(tos_file, "r") as f:
                    st.session_state.tos_text = json.load(f)
            else:
                st.session_state.tos_text = {}

        if current_domain in st.session_state.tos_text:
            render_tos_text(current_domain, st.session_state.tos_text[current_domain])


def annotation_viewer():
    st.header("View Annotations")
    schema = load_annotation_schema()

    # Show current user
    st.write(f"**Current User:** {st.session_state.current_user}")

    # Domain set selection - filter for current user
    domain_sets_options = [
        f for f in list(ANNOTATIONS_DIR.glob("*.json")) if "annotations" in f.stem
    ]
    domain_set_to_view = st.selectbox(
        "Select domain set",
        options=domain_sets_options,
        format_func=lambda x: x.stem,
        key="domain_set_to_view",
    )

    # Load domains
    annotation_data = json.load(open(domain_set_to_view, "r"))
    schema = load_annotation_schema()

    # Create summary statistics for each field
    st.write("### Annotation Summary")

    for field in schema.fields:
        st.write(f"#### {field.name}")

        # Get all values for this field
        field_values = [
            ann.get(field.id) for ann in annotation_data.values() if field.id in ann
        ]
        total_annotations = len(field_values)

        if total_annotations == 0:
            st.write("No annotations yet")
            continue

        if field.type == FieldType.BOOLEAN:
            true_count = sum(1 for v in field_values if v)
            st.write(f"True: {true_count} ({true_count/total_annotations:.1%})")
            st.write(
                f"False: {total_annotations - true_count} ({(total_annotations-true_count)/total_annotations:.1%})"
            )

        elif field.type == FieldType.SINGLE_SELECT:
            value_counts = {}
            for value in field_values:
                value_counts[value] = value_counts.get(value, 0) + 1

            for option in field.options or []:
                count = value_counts.get(option, 0)
                st.write(f"{option}: {count} ({count/total_annotations:.1%})")

        elif field.type == FieldType.MULTI_SELECT:
            option_counts = {}
            for values in field_values:
                if values:
                    for value in values:
                        option_counts[value] = option_counts.get(value, 0) + 1

            for option in field.options or []:
                count = option_counts.get(option, 0)
                st.write(f"{option}: {count} ({count/total_annotations:.1%})")

        elif field.type == FieldType.TEXT:
            pass

        elif field.type == FieldType.TEXTAREA:
            pass

        st.write("---")


if __name__ == "__main__":
    main()
